"""Shared, host-agnostic resolvers used by both layout renderers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from cellier.convenience.gui._controls_config import ChannelControlsConfig

if TYPE_CHECKING:
    from uuid import UUID

    from cellier.visuals._channel_appearance import ChannelAppearance


class ResolvedChannelControls(NamedTuple):
    """The data needed to build a channel-controls widget for one viewer."""

    config: ChannelControlsConfig
    visual_ids: list[UUID]
    channels: dict[int, ChannelAppearance]


def _resolve_channel_visual_ids(viewer: object) -> ResolvedChannelControls | None:
    """Resolve the configured channel visual(s) for *viewer*.

    Finds the first ``ChannelControlsConfig`` recorded on
    ``viewer._controls_configs`` and returns its config, the visual ids the
    channel widget should drive, and the channel appearances to seed it.

    For a single-panel ``Viewer`` the entry maps directly to one visual id.
    For an ``OrthoViewer`` the entry's key is a representative (first-panel)
    visual id and ``viewer._visual_groups`` maps it to the sibling
    visual ids across the four panels (design section 7.3).

    Returns ``None`` when no channel controls are configured.

    Raises
    ------
    ValueError
        When ``len(channels) > min(max_channels_2d, max_channels_3d)`` on the
        resolved visual -- over-cap channels would become silent render no-ops,
        so this fails loudly at widget-build time (design section 7.3 / 11.4).
    """
    controller = getattr(viewer, "controller", None)
    controls_configs: dict = getattr(viewer, "_controls_configs", {})
    if controller is None or not controls_configs:
        return None

    rep_id = None
    config = None
    for visual_id, cfg in controls_configs.items():
        if isinstance(cfg, ChannelControlsConfig):
            rep_id = visual_id
            config = cfg
            break
    if config is None:
        return None

    groups: dict | None = getattr(viewer, "_visual_groups", None)
    if groups is not None and rep_id in groups:
        visual_ids = list(groups[rep_id])
    else:
        visual_ids = [rep_id]

    visual = controller.get_visual_model(rep_id)
    channels = visual.channels

    cap = min(int(visual.max_channels_2d), int(visual.max_channels_3d))
    if len(channels) > cap:
        raise ValueError(
            f"Channel controls require len(channels) <= min(max_channels_2d, "
            f"max_channels_3d) = {cap}; got {len(channels)}. Raise the caps on "
            f"add_multichannel_image* if you need more simultaneous channels."
        )

    return ResolvedChannelControls(config, visual_ids, channels)


def channel_widget_kwargs(
    config: ChannelControlsConfig,
    channels: dict[int, ChannelAppearance],
) -> dict:
    """Build the toolkit-neutral kwargs shared by both channel widgets.

    ``QtChannelList`` and ``AnywidgetChannelList`` accept the same construction
    keywords (``clim_range``, ``colormap_names``, ``fields``,
    ``channel_labels``); this derives them from *config*, inferring
    ``clim_range`` from the channels' current clim when it is not configured.
    """
    kwargs: dict = {}
    if config.fields is not None:
        kwargs["fields"] = config.fields
    if config.colormap_names is not None:
        kwargs["colormap_names"] = config.colormap_names
    if config.channel_labels is not None:
        kwargs["channel_labels"] = config.channel_labels

    if config.clim_range is not None:
        kwargs["clim_range"] = config.clim_range
    elif channels:
        los = [float(ch.clim[0]) for ch in channels.values()]
        his = [float(ch.clim[1]) for ch in channels.values()]
        kwargs["clim_range"] = (min([*los, 0.0]), max([*his, 1.0]))

    return kwargs


# ── Appearance controls: the toolkit-neutral decision layer ──────────────────
#
# Design section 7.3.  Which controls a panel contains, in what order, seeded
# with what values, is one question with one answer -- it was previously
# answered twice, once per toolkit, in ~110 near-identical lines each
# (section 4.2).  These two functions are that answer, as pure data: no
# controller, no widgets, no toolkit import, so the whole decision layer is
# unit-testable with no fixtures.  Each renderer keeps only a dispatch table
# mapping ``ControlSpec.kind`` to its own widget class.


@dataclass(frozen=True)
class ControlSpec:
    """One control to build, described without reference to any toolkit.

    Parameters
    ----------
    kind : str
        Which control to build: ``color_map``, ``clim``, ``render``,
        ``lod_bias``, ``aabb`` or ``dataset_info``.  A renderer with no
        builder for a kind skips it -- that is how ``dataset_info`` stays
        anywidget-only without the Qt path needing to know it exists.
    title : str
        What the control is called, e.g. ``"Contrast limits"``.  Both front
        ends pass it to the widget, which draws it itself -- as a label beside
        a single control, or as a heading over a block of rows
        (``plans/label_ownership_unification.md``).
    values : dict[str, object]
        Construction keywords for the widget, already read off the model.
        Toolkit-neutral: a keyword only one toolkit accepts (Qt's
        ``dtype_max``) is derived inside that toolkit's builder from what is
        here.
    """

    kind: str
    title: str
    values: dict[str, object]


class AppearanceSpecs(NamedTuple):
    """The controls to build for one visual, plus what was asked for and dropped.

    Parameters
    ----------
    specs : list[ControlSpec]
        The controls, in display order.
    skipped : list[str]
        Requested field names that produced no control, either because the
        config class has no control for them or because the visual's
        appearance model does not carry them.  Reported rather than silently
        dropped (design section 9.5 step 4); stage 3 turns the first case
        into an error at config-construction time, which leaves this for the
        second.
    """

    specs: list[ControlSpec]
    skipped: list[str]


_CONTROL_TITLES = {
    "color_map": "Colormap",
    "clim": "Contrast limits",
    "render": "Render mode",
    "lod_bias": "LOD bias",
    "aabb": "Bounding box",
    "dataset_info": "Dataset info",
}
"""Titles for the control kinds that need one written out.

A kind not listed here is a plain single-field control, and its title is
derived from the field name -- so a config class can name a new kind without
registering anything.

Each of these is also a widget class's ``DEFAULT_TITLE`` on both toolkits,
pinned by ``test_composite_default_titles_match_the_shared_vocabulary``: this
is the name the renderers pass in, that is the name a directly-constructed
widget uses.
"""


def _default_title(field: str) -> str:
    """Title for a single-field control, e.g. ``wireframe_thickness``."""
    return field.replace("_", " ").capitalize()


def appearance_specs(visual: object, config: object) -> AppearanceSpecs:
    """Decide which appearance controls *visual* should get under *config*.

    Pure: reads the visual's appearance model and the config, and returns
    data.  Both renderers walk the result and build widgets from it.

    The requested field names are matched against the config class's
    ``APPEARANCE_CONTROLS`` map (design section 6.5.2 decision 6), so which
    widget a name means is decided by the kind of visual being configured
    rather than by the name alone.  Group order is the map's order, not the
    caller's -- the ``appearance`` docstring's "in display order" has never
    been true and stage 1 does not change that.

    Parameters
    ----------
    visual : BaseVisual
        The visual whose appearance the panel will drive.
    config : BaseControlsConfig
        The recorded controls config.

    Returns
    -------
    AppearanceSpecs
    """
    from cellier.convenience.gui._controls_config import (
        InMemoryImageControlsConfig,
        MultiscaleImageControlsConfig,
    )
    from cellier.gui._appearance_fields import (
        APPEARANCE_FIELD_WIDGETS,
        literal_choices,
    )
    from cellier.gui._colormap_util import colormap_to_str

    appearance = getattr(config, "appearance", False)
    if appearance is True:
        # "Use this config class's default panel" -- every field it can drive
        # (design section 11.2).  ``True`` was a dead value before stage 5:
        # the annotation admitted it and it took the same branch as ``False``.
        field_list = list(type(config).default_appearance_fields())
        explicit = False
    else:
        field_list = appearance if isinstance(appearance, list) and appearance else None
        explicit = True
    if not field_list or not hasattr(visual, "appearance"):
        return AppearanceSpecs([], list(field_list or []) if explicit else [])

    app = visual.appearance
    controls_map: dict[str, str] = type(config).APPEARANCE_CONTROLS
    requested = set(field_list)

    # A field is honoured when the config class has a control for it *and* the
    # visual's model actually carries it.  Both halves matter: a config can be
    # paired with a visual whose model is narrower (a multiscale config on an
    # in-memory image has no ``lod_bias`` to drive).
    honoured = {
        field for field in requested if field in controls_map and hasattr(app, field)
    }
    # A default list is "everything this config *can* drive", and the visual's
    # model is often narrower on purpose -- a flat mesh has no ``shininess``.
    # Reporting those would warn on every ``appearance=True`` panel, so only
    # fields the caller named are reported as dropped.
    skipped = sorted(requested - honoured) if explicit else []

    raw_clim = tuple(getattr(app, "clim", (0.0, 1.0)))
    if (
        isinstance(config, InMemoryImageControlsConfig)
        and config.clim_range is not None
    ):
        clim_range: tuple[float, float] = config.clim_range
    else:
        clim_range = (min(0.0, float(raw_clim[0])), max(1.0, float(raw_clim[1])))

    values_for = {
        "color_map": lambda: {
            "initial_colormap": colormap_to_str(getattr(app, "color_map", "grays")),
            "colormap_names": config.colormap_names
            if isinstance(config, InMemoryImageControlsConfig)
            else None,
        },
        "clim": lambda: {
            "clim_range": clim_range,
            "initial_clim": raw_clim,
        },
        "render": lambda: {
            "clim_range": clim_range,
            "initial_render_mode": getattr(app, "render_mode", "mip"),
            "initial_threshold": getattr(app, "iso_threshold", 0.2),
            "initial_attenuation": getattr(app, "attenuation", 1.0),
        },
        "lod_bias": lambda: {"initial_lod_bias": float(getattr(app, "lod_bias", 1.0))},
    }

    specs: list[ControlSpec] = []
    seen: set[str] = set()
    for field, kind in controls_map.items():
        if field not in honoured or kind in seen:
            continue
        seen.add(kind)
        # A kind with no bespoke entry is a plain single-field control: its
        # value is the field's current value, and its title comes from the
        # shared widget table.  The bespoke entries exist for the controls
        # that read several fields at once, or that need something the model
        # does not hold directly.
        if kind in values_for:
            values = values_for[kind]()
            title = _CONTROL_TITLES[kind]
        else:
            values = {"initial_value": getattr(app, field)}
            # A Literal field's options come from the model's own annotation,
            # so the in-memory and multiscale variants of a field each offer
            # their own set without a list restated here.
            choices = literal_choices(app, field)
            if choices:
                values["choices"] = choices
            stem_title = APPEARANCE_FIELD_WIDGETS.get(kind)
            title = stem_title[1] if stem_title else _default_title(field)
        specs.append(ControlSpec(kind, title, values))

    # The bounding box is not a field of the appearance model and is not
    # requested by name: ``aabb`` is on ``BaseVisual`` with a default factory,
    # so every visual has one and every configured panel gets the control.
    # This is what the anywidget path already did; stage 1 gives Qt the same
    # (design section 7.1).
    aabb = getattr(visual, "aabb", None)
    if aabb is not None:
        specs.append(
            ControlSpec(
                "aabb",
                _CONTROL_TITLES["aabb"],
                {
                    "initial_enabled": aabb.enabled,
                    "initial_line_width": aabb.line_width,
                    "initial_color": aabb.color,
                },
            )
        )

    dataset_info = (
        config.dataset_info if isinstance(config, MultiscaleImageControlsConfig) else ""
    )
    if dataset_info:
        specs.append(
            ControlSpec(
                "dataset_info",
                _CONTROL_TITLES["dataset_info"],
                {"html": dataset_info},
            )
        )

    return AppearanceSpecs(specs, skipped)


def warn_skipped_appearance_fields(
    skipped: list[str], visual: object, config: object
) -> None:
    """Warn about requested fields that produced no control.

    Stage 3 makes a name the *config class* cannot drive raise at
    construction, which leaves one residual case this cannot catch: a name
    valid for the config class whose value the particular *visual*'s
    appearance model does not carry -- a ``MultiscaleImageControlsConfig``
    paired with an in-memory image, say.  That is still a silent drop unless
    something says so, and it is the renderer's to surface because only the
    renderer has both halves (design section 9.5 step 4).
    """
    if not skipped:
        return
    warnings.warn(
        f"{type(config).__name__} requested appearance field(s) "
        f"{skipped} that {type(visual).__name__}'s appearance model does not "
        f"have; no control was built for them.",
        UserWarning,
        stacklevel=3,
    )


class AppearanceTarget(NamedTuple):
    """What an appearance dock drives.

    Parameters
    ----------
    visual : BaseVisual
        The representative visual.  Its appearance model is what the controls
        are seeded from; on an ``OrthoViewer`` it is the first panel's visual
        and the other three are guaranteed equal to it.
    config : BaseControlsConfig
        The recorded controls config.
    visual_ids : list[UUID]
        Every visual the controls write to -- one on a ``Viewer``, the four
        panel siblings on an ``OrthoViewer``.  The widgets accept this
        directly (see ``cellier.gui._appearance_fields.VisualIdGroup``), so
        the spec walk is identical either way and only the id list differs.
    """

    visual: object
    config: object
    visual_ids: list


def select_appearance_target(viewer: object) -> AppearanceTarget | None:
    """Find the visual(s) whose appearance controls a dock should render.

    First-match-wins in visual order, matching the pre-stage-1 behaviour
    exactly -- supporting two independently configured visuals is design
    section 4.4 and is deferred.  Channel configs are skipped: they are
    resolved by :func:`_resolve_channel_visual_ids` instead.

    **Multi-scene aware.**  A ``Viewer`` exposes ``scene``; an ``OrthoViewer``
    exposes ``scenes`` and nothing else, which is why ``AppearanceControls()``
    used to be a silent no-op on it -- both renderers read ``viewer.scene``,
    got ``None``, and produced no dock and no error (section 4.1).  Walking
    ``scenes`` and expanding the group through ``_visual_groups`` is the whole
    fix; the renderers are unchanged.

    Returns ``None`` when the viewer has no scenes or nothing is configured.
    """
    controls_configs: dict = getattr(viewer, "_controls_configs", {})
    if not controls_configs:
        return None

    scene = getattr(viewer, "scene", None)
    if scene is not None:
        scenes = [scene]
    else:
        scenes_map = getattr(viewer, "scenes", None)
        if not scenes_map:
            return None
        scenes = list(scenes_map.values())

    groups: dict = getattr(viewer, "_visual_groups", {}) or {}
    for candidate_scene in scenes:
        for visual in candidate_scene.visuals:
            config = controls_configs.get(visual.id)
            if config is None or isinstance(config, ChannelControlsConfig):
                continue
            visual_ids = list(groups.get(visual.id, [visual.id]))
            return AppearanceTarget(visual, config, visual_ids)
    return None
