"""Utilities for creating and processing slice data."""

from cellier.models.scene import DimsManager
from cellier.models.visuals.base import BaseVisual
from cellier.slicer.world_slice import (
    AxisAligned2DWorldSlice,
    AxisAligned3DWorldSlice,
    BaseWorldSlice,
)
from cellier.types import SelectedRegion


def world_slice_from_dims_manager(dims_manager: DimsManager) -> BaseWorldSlice:
    """Construct a world slice from the current dims state.

    Parameters
    ----------
    dims_manager: DimsManager
        THe dimension manager from which to construct the world slice.

    Returns
    -------
    world_slice : BaseWorldSlice
        The constructed world slice.
    """
    if dims_manager.ndisplay == 2:
        # 2D world slice
        return AxisAligned2DWorldSlice.from_dims(dims_manager)

    elif dims_manager.ndisplay == 3:
        # 3D world slice
        return AxisAligned3DWorldSlice.from_dims(dims_manager)

    else:
        raise ValueError("Unsupported DimsManager state: {DimsManager}")


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
