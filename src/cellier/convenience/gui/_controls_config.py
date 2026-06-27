"""Visual-type-specific controls configuration dataclasses.

These are convenience-layer objects only -- they are not part of the core
cellier model and are not serialized.  Each class mirrors the fields of its
corresponding *ControlsKwargs TypedDict so that callers can pass either a
plain dict or a pre-built config instance to Viewer.add_* methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cellier.convenience._kwarg_dicts import (
        InMemoryImageControlsKwargs,
        MultiscaleImageControlsKwargs,
    )


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


def resolve_inmemory_image_controls(
    controls: InMemoryImageControlsConfig | InMemoryImageControlsKwargs | None,
) -> InMemoryImageControlsConfig | None:
    """Resolve a dict or config instance to ``InMemoryImageControlsConfig``."""
    if controls is None:
        return None
    if isinstance(controls, dict):
        return InMemoryImageControlsConfig(**controls)
    return controls


def resolve_multiscale_image_controls(
    controls: MultiscaleImageControlsConfig | MultiscaleImageControlsKwargs | None,
) -> MultiscaleImageControlsConfig | None:
    """Resolve a dict or config instance to ``MultiscaleImageControlsConfig``."""
    if controls is None:
        return None
    if isinstance(controls, dict):
        return MultiscaleImageControlsConfig(**controls)
    return controls
