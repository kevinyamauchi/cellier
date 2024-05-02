"""Models for mesh data_stores streams."""

from abc import ABC
from typing import List, Literal

from cellier.models.data_stores.mesh import MeshDataStoreSlice
from cellier.models.data_streams.base_data_stream import BaseDataStream
from cellier.slicer.data_slice import DataSliceRequest


class BaseMeshDataStream(BaseDataStream, ABC):
    """Base class for all mesh data_stores streams."""

    pass


class MeshSynchronousDataStream(BaseMeshDataStream):
    """Class for synchronous mesh data_stores streams."""

    data_store_id: str
    selectors: List[str]

    # this is used for a discriminated union
    stream_type: Literal["mesh_synchronous"] = "mesh_synchronous"

    def get_data_store_slice(
        self, slice_request: DataSliceRequest
    ) -> MeshDataStoreSlice:
        """Get slice object to get the requested world data slice from the data store.

        todo: handle mixed dimensions, etc.

        Parameters
        ----------
        slice_request : DataSliceRequest
            The requested data slice to generate the data store slice from.
        """
        return MeshDataStoreSlice(
            scene_id=slice_request.scene_id,
            visual_id=slice_request.visual_id,
            resolution_level=slice_request.resolution_level,
            displayed_dimensions=slice_request.world_slice.displayed_dimensions,
        )
