"""Unit tests for the toolkit-neutral appearance decision layer.

Stage 1 of ``plans/convenience_cleanup.md`` (section 7.3) moved the "which
controls, in what order, with what values" question out of the two renderers
and into ``layout._shared.appearance_specs``, which is **pure** -- no
controller, no widgets, no toolkit import.  That is the coverage win the
refactor was for: the field-predicate matrix and the clim-range inference were
previously reachable only through a Qt or an anywidget fixture, and each was
tested against only one of the two copies (section 4.2).

Nothing here constructs a widget or a viewer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from cellier.convenience.gui._controls_config import (
    BaseControlsConfig,
    ChannelControlsConfig,
    InMemoryImageControlsConfig,
    MultiscaleImageControlsConfig,
)
from cellier.convenience.layout._shared import (
    appearance_specs,
    select_appearance_target,
)
from cellier.visuals._base_visual import AABBParams
from cellier.visuals._image import MultiscaleImageAppearance
from cellier.visuals._image_memory import InMemoryImageAppearance


class _FakeVisual:
    """The two attributes ``appearance_specs`` reads, and an id."""

    def __init__(self, appearance, aabb=None, visual_id="v0"):
        self.appearance = appearance
        self.aabb = AABBParams() if aabb is None else aabb
        self.id = visual_id


def _in_memory(**kwargs) -> _FakeVisual:
    return _FakeVisual(InMemoryImageAppearance(color_map="grays", **kwargs))


def _multiscale(**kwargs) -> _FakeVisual:
    return _FakeVisual(MultiscaleImageAppearance(color_map="viridis", **kwargs))


def kinds(result) -> list[str]:
    """The spec kinds, in order."""
    return [spec.kind for spec in result.specs]


# ── the field predicate matrix ───────────────────────────────────────────────


def test_every_multiscale_field_maps_to_a_control():
    result = appearance_specs(
        _multiscale(),
        MultiscaleImageControlsConfig(
            appearance=[
                "color_map",
                "clim",
                "render_mode",
                "iso_threshold",
                "attenuation",
                "lod_bias",
            ]
        ),
    )
    assert kinds(result) == ["color_map", "clim", "render", "lod_bias", "aabb"]
    assert result.skipped == []


def test_the_three_render_fields_collapse_into_one_control():
    """``render_mode``/``iso_threshold``/``attenuation`` share one widget."""
    for fields in (
        ["render_mode"],
        ["iso_threshold"],
        ["render_mode", "iso_threshold", "attenuation"],
    ):
        result = appearance_specs(
            _multiscale(), MultiscaleImageControlsConfig(appearance=fields)
        )
        assert kinds(result) == ["render", "aabb"], fields


def test_order_is_the_config_maps_order_not_the_callers():
    """Group order follows ``APPEARANCE_CONTROLS``, not the ``appearance`` list.

    The ``appearance`` docstring says "in display order"; it never has been.
    Pinned here as well as in the Qt acceptance test so a change shows up
    without a toolkit fixture.
    """
    result = appearance_specs(
        _in_memory(), InMemoryImageControlsConfig(appearance=["clim", "color_map"])
    )
    assert kinds(result) == ["color_map", "clim", "aabb"]


# A field the config class does not know -- a typo, or a real name that does
# not apply to this config -- no longer reaches this function: stage 3 rejects
# it in ``__post_init__``.  See ``test_controls_config_validation.py``.


def test_a_field_missing_from_the_visuals_model_is_skipped():
    """A multiscale config on an in-memory visual: the model is narrower.

    The config class knows ``lod_bias``; the model does not carry it.  Both
    halves of the predicate have to hold.
    """
    result = appearance_specs(
        _in_memory(), MultiscaleImageControlsConfig(appearance=["clim", "lod_bias"])
    )
    assert kinds(result) == ["clim", "aabb"]
    assert result.skipped == ["lod_bias"]


@pytest.mark.parametrize("appearance", [False, []])
def test_no_field_list_yields_no_specs(appearance):
    """``False`` and ``[]`` produce nothing -- not even the AABB."""
    result = appearance_specs(
        _in_memory(), InMemoryImageControlsConfig(appearance=appearance)
    )
    assert result.specs == []


def test_appearance_true_resolves_to_the_config_classes_default_list():
    """Stage 5 gave the dead ``True`` value the obvious meaning."""
    result = appearance_specs(
        _in_memory(), InMemoryImageControlsConfig(appearance=True)
    )
    assert kinds(result) == [
        "visible",
        "opacity",
        "color_map",
        "clim",
        "render",
        "aabb",
    ]


def test_a_default_list_does_not_report_fields_the_model_lacks():
    """``True`` means "everything this config can drive", and models narrow it.

    A flat mesh has no ``shininess``.  Warning about that on every default
    panel would be noise, so only fields the caller *named* are reported as
    skipped.
    """
    from cellier.convenience.gui._controls_config import MeshControlsConfig
    from cellier.visuals._mesh_memory import MeshFlatAppearance

    visual = _FakeVisual(MeshFlatAppearance())

    assert appearance_specs(visual, MeshControlsConfig(appearance=True)).skipped == []
    assert appearance_specs(
        visual, MeshControlsConfig(appearance=["shininess"])
    ).skipped == ["shininess"]


def test_a_visual_with_no_appearance_yields_no_specs():
    class _NoAppearance:
        id = "v0"
        aabb = AABBParams()

    result = appearance_specs(
        _NoAppearance(), InMemoryImageControlsConfig(appearance=["color_map"])
    )
    assert result.specs == []


# ── the values carried on each spec ──────────────────────────────────────────


def test_clim_range_is_inferred_from_the_current_clim():
    """Widened to include 0 and 1, matching what both builders used to do."""
    result = appearance_specs(
        _in_memory(clim=(-5.0, 200.0)),
        InMemoryImageControlsConfig(appearance=["clim"]),
    )
    (clim_spec, _aabb) = result.specs
    assert clim_spec.values["clim_range"] == (-5.0, 200.0)
    assert clim_spec.values["initial_clim"] == (-5.0, 200.0)


def test_a_clim_inside_the_unit_interval_is_widened_to_it():
    result = appearance_specs(
        _in_memory(clim=(0.25, 0.75)),
        InMemoryImageControlsConfig(appearance=["clim"]),
    )
    assert result.specs[0].values["clim_range"] == (0.0, 1.0)


def test_a_configured_clim_range_wins_over_the_inferred_one():
    result = appearance_specs(
        _in_memory(clim=(0.0, 1.0)),
        InMemoryImageControlsConfig(appearance=["clim"], clim_range=(0.0, 4095.0)),
    )
    assert result.specs[0].values["clim_range"] == (0.0, 4095.0)


def test_the_render_spec_carries_clim_range_for_qts_dtype_max():
    """Qt derives ``dtype_max`` from this; the anywidget builder ignores it.

    A toolkit-specific keyword derived inside that toolkit's builder is the
    seam's escape hatch for exactly this kind of asymmetry (section 7.3).
    """
    result = appearance_specs(
        _multiscale(clim=(0.0, 65535.0)),
        MultiscaleImageControlsConfig(appearance=["render_mode"]),
    )
    assert result.specs[0].values["clim_range"] == (0.0, 65535.0)


def test_colormap_names_come_from_the_config():
    result = appearance_specs(
        _in_memory(),
        InMemoryImageControlsConfig(
            appearance=["color_map"], colormap_names=["magma", "grays"]
        ),
    )
    assert result.specs[0].values["colormap_names"] == ["magma", "grays"]
    # Normalised through ``colormap_to_str``, which is what the widgets take.
    assert result.specs[0].values["initial_colormap"] == "colorbrewer:greys"


def test_the_aabb_spec_is_seeded_from_the_visual_not_from_defaults():
    visual = _in_memory()
    visual.aabb = AABBParams(enabled=True, line_width=7.5, color="#ff00ff")

    result = appearance_specs(
        visual, InMemoryImageControlsConfig(appearance=["color_map"])
    )
    aabb_spec = result.specs[-1]
    assert aabb_spec.kind == "aabb"
    assert aabb_spec.values == {
        "initial_enabled": True,
        "initial_line_width": 7.5,
        "initial_color": "#ff00ff",
    }


def test_dataset_info_is_appended_last_and_only_when_non_empty():
    config = MultiscaleImageControlsConfig(
        appearance=["color_map"], dataset_info=[("Scale levels", "4")]
    )
    result = appearance_specs(_multiscale(), config)
    assert kinds(result) == ["color_map", "aabb", "dataset_info"]
    assert result.specs[-1].values == {"rows": [("Scale levels", "4")]}

    config.dataset_info = ()
    assert kinds(appearance_specs(_multiscale(), config)) == ["color_map", "aabb"]


def test_dataset_info_rows_are_coerced_to_strings():
    """A value read off a store is rarely already a string.

    Both front ends display what they are given verbatim, so the neutral layer
    is where an int shape or a numpy dtype becomes text -- once, rather than
    once per toolkit.
    """
    config = MultiscaleImageControlsConfig(
        appearance=["color_map"], dataset_info=[("Scale levels", 4)]
    )
    result = appearance_specs(_multiscale(), config)
    assert result.specs[-1].values == {"rows": [("Scale levels", "4")]}


def test_titles_are_shared_by_both_front_ends():
    """One title per kind, defined once -- the Qt group box reads these."""
    result = appearance_specs(
        _multiscale(),
        MultiscaleImageControlsConfig(
            appearance=["color_map", "clim", "render_mode", "lod_bias"]
        ),
    )
    assert [spec.title for spec in result.specs] == [
        "Colormap",
        "Contrast limits",
        "Render mode",
        "LOD bias",
        "Bounding box",
    ]


# ── the per-config-class dispatch (design section 6.5.2 decision 6) ──────────


def test_the_same_field_name_can_mean_different_controls_per_config():
    """``render_mode`` is not one control; the config class decides.

    The image models spell it ``mip``/``iso``/``minip`` and the labels models
    ``iso_categorical``/``flat_categorical``.  A global field-name table would
    hand a labels visual the volume-render widget; dispatching on the config
    makes that impossible rather than merely unlikely.
    """

    @dataclass
    class _FakeLabelsConfig(BaseControlsConfig):
        APPEARANCE_CONTROLS: ClassVar[dict[str, str]] = {"render_mode": "labels_render"}

    from cellier.visuals._label_memory import InMemoryLabelsAppearance

    labels_visual = _FakeVisual(InMemoryLabelsAppearance())
    image_result = appearance_specs(
        _in_memory(), InMemoryImageControlsConfig(appearance=["render_mode"])
    )
    labels_result = appearance_specs(
        labels_visual, _FakeLabelsConfig(appearance=["render_mode"])
    )

    assert kinds(image_result)[0] == "render"
    assert kinds(labels_result)[0] == "labels_render"
    # A single-field kind carries the field's value, plus the Literal's own
    # options where it has them -- which is how one labels config serves both
    # the in-memory and the multiscale render-mode vocabularies.
    assert labels_result.specs[0].values == {
        "initial_value": "iso_categorical",
        "choices": ("iso_categorical", "flat_categorical"),
    }


# ── select_appearance_target ─────────────────────────────────────────────────


class _FakeScene:
    def __init__(self, visuals):
        self.visuals = visuals


class _FakeViewer:
    def __init__(self, scene, configs):
        self.scene = scene
        self._controls_configs = configs


def test_select_returns_the_first_configured_visual():
    first, second = _FakeVisual(None, visual_id="a"), _FakeVisual(None, visual_id="b")
    config_a = InMemoryImageControlsConfig(appearance=["clim"])
    config_b = InMemoryImageControlsConfig(appearance=["color_map"])
    viewer = _FakeViewer(_FakeScene([first, second]), {"a": config_a, "b": config_b})

    target = select_appearance_target(viewer)

    # First-match-wins in scene.visuals order, preserved exactly from before
    # the refactor; supporting two configured visuals is section 4.4.
    assert target.visual is first
    assert target.config is config_a


def test_select_skips_visuals_with_no_config():
    unconfigured, configured = (
        _FakeVisual(None, visual_id="a"),
        _FakeVisual(None, visual_id="b"),
    )
    config = InMemoryImageControlsConfig(appearance=["clim"])
    viewer = _FakeViewer(_FakeScene([unconfigured, configured]), {"b": config})

    assert select_appearance_target(viewer).visual is configured


def test_select_skips_channel_configs():
    """Channel controls are resolved by ``_resolve_channel_visual_ids``."""
    visual = _FakeVisual(None, visual_id="a")
    viewer = _FakeViewer(
        _FakeScene([visual]), {"a": ChannelControlsConfig(fields=["visible"])}
    )

    assert select_appearance_target(viewer) is None


def test_select_returns_none_without_a_scene_or_a_config():
    visual = _FakeVisual(None, visual_id="a")
    assert select_appearance_target(_FakeViewer(_FakeScene([visual]), {})) is None
    assert select_appearance_target(_FakeViewer(None, {"a": object()})) is None
    assert select_appearance_target(object()) is None


# ── dataset_info: the three forms of the setting ─────────────────────────────


class _FakeStore:
    """A minimal store: the one method ``dataset_info=True`` calls."""

    def __init__(self, info):
        self._info = info

    def dataset_info(self):
        return self._info


def test_dataset_info_true_asks_the_store_to_describe_itself():
    """The point of the generalization: rows come from the store, not the caller.

    Before this, every caller hand-wrote the rows, which is why both example
    scripts asserted a hardcoded ``"2x isotropic"`` that no longer matched an
    anisotropic pyramid.
    """
    from cellier.data._dataset_info import DatasetInfo, RowSection

    info = DatasetInfo(sections=[RowSection(None, [("Points", "12")])])
    result = appearance_specs(
        _multiscale(),
        MultiscaleImageControlsConfig(appearance=["color_map"], dataset_info=True),
        _FakeStore(info),
    )
    assert kinds(result) == ["color_map", "aabb", "dataset_info"]
    assert result.specs[-1].values == {"info": info}


def test_dataset_info_true_without_a_store_builds_no_block():
    """A block asserting that a store has no metadata is worse than no block."""
    config = MultiscaleImageControlsConfig(appearance=["color_map"], dataset_info=True)
    assert kinds(appearance_specs(_multiscale(), config)) == ["color_map", "aabb"]


def test_dataset_info_accepts_a_prebuilt_dataset_info():
    from cellier.data._dataset_info import DatasetInfo, RowSection

    info = DatasetInfo(sections=[RowSection(None, [("Nodes", "3")])])
    result = appearance_specs(
        _multiscale(),
        MultiscaleImageControlsConfig(appearance=["color_map"], dataset_info=info),
    )
    assert result.specs[-1].values == {"info": info}


def test_dataset_info_is_available_on_every_config_class():
    """It used to live on ``MultiscaleImageControlsConfig`` alone.

    That gate meant a points or graph visual could not show the block at all,
    however well its store could describe itself.
    """
    from cellier.convenience.gui._controls_config import (
        GraphControlsConfig,
        LinesControlsConfig,
        MeshControlsConfig,
        PointsControlsConfig,
    )
    from cellier.data._dataset_info import DatasetInfo, RowSection

    info = DatasetInfo(sections=[RowSection(None, [("Points", "12")])])
    for config_class in (
        PointsControlsConfig,
        LinesControlsConfig,
        MeshControlsConfig,
        GraphControlsConfig,
        BaseControlsConfig,
    ):
        config = config_class(appearance=["visible"], dataset_info=True)
        result = appearance_specs(_multiscale(), config, _FakeStore(info))
        assert "dataset_info" in kinds(result), config_class.__name__
