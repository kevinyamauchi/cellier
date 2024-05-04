"""Utilities for creating and processing slice data."""

from cellier.models.scene import DimsManager
from cellier.slicer.world_slice import (
    AxisAligned2DWorldSlice,
    AxisAligned3DWorldSlice,
    BaseWorldSlice,
)


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
