"""Classes for Points Data Streams."""

from abc import ABC
from typing import List, Literal

from cellier.models.data_stores.points import PointDataStoreSlice
from cellier.models.data_streams.base_data_stream import BaseDataStream
from cellier.slicer.data_slice import DataSliceRequest


class BasePointDataStream(BaseDataStream, ABC):
    """Base class for all mesh data_stores streams."""

    pass


class PointsSynchronousDataStream(BasePointDataStream):
    """Class for synchronous mesh data_stores streams."""

    data_store_id: str
    selectors: List[str]

    # this is used for a discriminated union
    stream_type: Literal["points_synchronous"] = "points_synchronous"

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
            point=slice_request.world_slice.point,
            margin_negative=slice_request.world_slice.margin_negative,
            margin_positive=slice_request.world_slice.margin_positive,
        )
