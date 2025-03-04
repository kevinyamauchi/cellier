"""Infrastructure to manage slicing of the data for viewing."""

from cellier.slicer.slicer import SynchronousDataSlicer
from cellier.slicer.world_slice import ObliqueWorldSlice

__all__ = ["SynchronousDataSlicer", "ObliqueWorldSlice"]
