"""Shared, host-agnostic resolvers used by both layout renderers."""

from __future__ import annotations

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
    visual id and ``viewer._channel_visual_groups`` maps it to the sibling
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

    groups: dict | None = getattr(viewer, "_channel_visual_groups", None)
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

    ``QtChannelList`` and ``ChannelPanel`` accept the same construction
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
