"""Infrastructure to manage slicing of the data for viewing."""

from cellier.slicer.data_slice import DataSliceRequest
from cellier.slicer.transforms import AffineTransform
from cellier.slicer.world_slice import AxisAligned2DWorldSlice, AxisAligned3DWorldSlice

__all__ = [
    "DataSliceRequest",
    "AxisAligned2DWorldSlice",
    "AxisAligned3DWorldSlice",
    "AffineTransform",
]
