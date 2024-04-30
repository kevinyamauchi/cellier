"""Classes for Points Data Streams."""

from abc import ABC
from typing import List

from cellier.models.data_stores.points import PointDataStoreSlice
from cellier.models.data_streams.base_data_stream import BaseDataStream
from cellier.slicer.data_slice import DataSliceRequest


class BasePointDataStream(BaseDataStream, ABC):
    """Base class for all mesh data_stores streams."""

    pass


class PointynchronousDataStream(BasePointDataStream):
    """Class for synchronous mesh data_stores streams."""

    data_store_id: str
    selectors: List[str]

    def get_data_store_slice(
        self, slice_request: DataSliceRequest
    ) -> PointDataStoreSlice:
        """Get slice object to get the requested world data slice from the data store.

        todo: handle mixed dimensions, etc.

        Parameters
        ----------
        slice_request : DataSliceRequest
            The requested data slice to generate the data store slice from.
        """
        return PointDataStoreSlice(
            scene_id=slice_request.scene_id,
            visual_id=slice_request.visual_id,
            resolution_level=slice_request.resolution_level,
            displayed_dimensions=slice_request.world_slice.displayed_dimensions,
        )
