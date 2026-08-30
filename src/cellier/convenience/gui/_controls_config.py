"""Visual-type-specific controls configuration dataclasses.

These are convenience-layer objects only -- they are not part of the core
cellier model and are not serialized.  Pass an instance to the ``controls=``
argument of the ``Viewer`` / ``OrthoViewer`` ``add_*`` methods.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseControlsConfig:
    """Base controls configuration shared by all visual types.

    Parameters
    ----------
    appearance : list[str] or False
        Fields to show in the appearance panel, in display order.
        ``False`` (default) hides the appearance panel entirely.
    """

    appearance: list[str] | bool = False


@dataclass
class InMemoryImageControlsConfig(BaseControlsConfig):
    """Controls configuration for in-memory image visuals.

    Parameters
    ----------
    appearance : list[str] or False
        Appearance fields in display order, e.g.
        ``["color_map", "clim", "render_mode", "iso_threshold"]``.
    colormap_names : list[str] or None
        Names available in the colormap dropdown.  Defaults to a curated
        list when ``None``.
    clim_range : tuple[float, float] or None
        ``(min, max)`` bounds for the contrast-limits slider.  Inferred
        from the visual's current clim when ``None``.
    """

    colormap_names: list[str] | None = None
    clim_range: tuple[float, float] | None = None


@dataclass
class MultiscaleImageControlsConfig(InMemoryImageControlsConfig):
    """Controls configuration for multiscale image visuals.

    Parameters
    ----------
    appearance : list[str] or False
        Appearance fields in display order, e.g.
        ``["color_map", "clim", "render_mode", "iso_threshold",
        "attenuation", "lod_bias"]``.
    colormap_names : list[str] or None
        Names available in the colormap dropdown.
    clim_range : tuple[float, float] or None
        ``(min, max)`` bounds for the contrast-limits slider.
    dataset_info : str
        Pre-formatted HTML for the dataset-info detail block.
        Empty string hides the block.
    """

    dataset_info: str = ""


@dataclass
class ChannelControlsConfig(BaseControlsConfig):
    """Controls configuration for multichannel image visuals.

    Parameters
    ----------
    fields : list[str] or None
        Per-channel fields to expose, in display order.  Defaults to
        ``["visible", "color_map", "clim", "opacity"]`` when ``None``.
    colormap_names : list[str] or None
        Names available in each channel's colormap control.  Defaults to a
        curated list when ``None``.
    clim_range : tuple[float, float] or None
        ``(min, max)`` bounds for the contrast-limits sliders.  Inferred from
        the channels' current clim when ``None``.
    channel_labels : dict[int, str] or None
        Optional per-channel display labels keyed by channel index.  Defaults
        to ``"Channel {i}"`` when ``None``.
    """

    fields: list[str] | None = None
    colormap_names: list[str] | None = None
    clim_range: tuple[float, float] | None = None
    channel_labels: dict[int, str] | None = None
