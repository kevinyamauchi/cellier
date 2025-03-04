"""Classes and functions for working with data slices."""

from dataclasses import dataclass

import numpy as np

from cellier.slicer.transforms import BaseTransform
from cellier.slicer.world_slice import BaseWorldSlice


@dataclass(frozen=True)
class DataSliceRequest:
    """The data required to make a slice request.

    Note: this is an immutable dataclass.

    Parameters
    ----------
    world_slice : BaseWorldSlice
        The data of the world slice to get from the data stream.
        The coordinates are in world coordinates.
    visual_id : str
        The unique identifier for which visual this data slice
        will be sent to.
    scene_id : str
        The unique identifier for which scene this visual belongs to.
    request_id : str
        The unique identifier for this request.
    data_to_world_transform : BaseTransform
        The transformation from the data coordinates to the world coordinates.

    Attributes
    ----------
    world_slice : BaseWorldSlice
        The data of the world slice to get from the data stream.
        The coordinates are in world coordinates.
    resolution_level : int
        The resolution level to render where 0 is the highest resolution
        and high numbers correspond with more down sampling.
    data_store_id : str
        The ID of the datastore from which to get the data slice.
    visual_id : str
        The unique identifier for which visual this data slice
        will be sent to.
    scene_id : str
        The unique identifier for which scene this visual belongs to.
    request_id : str
        The unique identifier for this request.
    data_to_world_transform : BaseTransform
        The transformation from the data coordinates to the world coordinates.
    """

    world_slice: BaseWorldSlice
    resolution_level: int
    data_store_id: str
    scene_id: str
    visual_id: str
    request_id: str
    data_to_world_transform: BaseTransform


@dataclass(frozen=True)
class RenderedSliceData:
    """Base class for rendered slice data classes.

    Note: all data should be in data coordinates.
    They will be transformed into world coordinates by
    the visual.

    Attributes
    ----------
    scene_id : str
        The unique identifier for which scene this visual belongs to.
    visual_id : str
        The UID of the visual to be updated.
    resolution_level : int
        The resolution level to be rendered. 0 is the highest resolution
        and larger numbers are lower resolution.
    """

    scene_id: str
    visual_id: str
    resolution_level: int


@dataclass(frozen=True)
class RenderedMeshDataSlice(RenderedSliceData):
    """Data class for rendered mesh slice data.

    Attributes
    ----------
    scene_id : str
        The unique identifier for which scene this visual belongs to.
    visual_id : str
        The UID of the visual to be updated.
    resolution_level : int
        The resolution level to be rendered. 0 is the highest resolution
        and larger numbers are lower resolution.
    vertices : np.ndarray
        The vertex coordinates of the new slice.
    faces  : np.ndarray
        The face indices of the new slices.

    """

    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class RenderedPointsDataSlice(RenderedSliceData):
    """Data class for rendered points slice data.

    Attributes
    ----------
    scene_id : str
        The unique identifier for which scene this visual belongs to.
    visual_id : str
        The UID of the visual to be updated.
    resolution_level : int
        The resolution level to be rendered. 0 is the highest resolution
        and larger numbers are lower resolution.
    coordinates : np.ndarray
        The point coordinates of the new slice.
    """

    coordinates: np.ndarray


@dataclass(frozen=True)
class RenderedLinesDataSlice(RenderedSliceData):
    """Data class for rendered points slice data.

    Attributes
    ----------
    scene_id : str
        The unique identifier for which scene this visual belongs to.
    visual_id : str
        The UID of the visual to be updated.
    resolution_level : int
        The resolution level to be rendered. 0 is the highest resolution
        and larger numbers are lower resolution.
    coordinates : np.ndarray
        The point coordinates of the new lines slice.
    """

    coordinates: np.ndarray


@dataclass(frozen=True)
class RenderedImageDataSlice(RenderedSliceData):
    """Data class for rendered image slice data.

    Attributes
    ----------
    scene_id : str
        The unique identifier for which scene this visual belongs to.
    visual_id : str
        The UID of the visual to be updated.
    resolution_level : int
        The resolution level to be rendered. 0 is the highest resolution
        and larger numbers are lower resolution.
    data : np.ndarray
        The point coordinates of the new slice.
    texture_start_index : tuple[int, int, int]
        The index of the minimum index to insert the chunk into.
    """

    data: np.ndarray
    texture_start_index: tuple[int, int, int]
