"""Utilities for creating and processing slice data."""

from cellier.models.scene import DimsManager
from cellier.models.visuals.base import BaseVisual
from cellier.types import SelectedRegion


def world_selected_region_from_dims(
    dims_manager: DimsManager, visual: BaseVisual
) -> SelectedRegion:
    """Construct a world sample from the current dims state.

    Parameters
    ----------
    dims_manager: DimsManager
        The dimension manager from which to construct the world selected region.
    visual : BaseVisual
        The visual object to use for sampling.

    Returns
    -------
    world_sample : SelectedRegion
        The constructed world sample.
    """
    if isinstance(visual, BaseVisual):
        return dims_manager.selection.to_state()
    else:
        raise TypeError(f"Unsupported visual type: {visual}")
