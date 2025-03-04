"""Models for Image data streams."""

from abc import ABC
from typing import List, Literal

from cellier.models.data_stores.image import ImageDataStoreSlice
from cellier.models.data_streams.base_data_stream import BaseDataStream
from cellier.slicer.data_slice import DataSliceRequest
from cellier.util.chunk import ImageDataStoreChunk


class BaseImageDataStream(BaseDataStream, ABC):
    """Base class for all mesh data_stores streams."""

    pass


class ImageSynchronousDataStream(BaseImageDataStream):
    """Class for synchronous image data_stores streams."""

    data_store_id: str
    selectors: List[str]

    # this is used for a discriminated union
    stream_type: Literal["image_synchronous"] = "image_synchronous"

    def get_data_store_slice(
        self, slice_request: DataSliceRequest
    ) -> ImageDataStoreSlice:
        """Get slice object to get the requested world data slice from the data store.

        todo: handle mixed dimensions, etc.

        Parameters
        ----------
        slice_request : DataSliceRequest
            The requested data slice to generate the data store slice from.
        """
        return ImageDataStoreSlice(
            scene_id=slice_request.scene_id,
            visual_id=slice_request.visual_id,
            resolution_level=slice_request.resolution_level,
            displayed_dimensions=slice_request.world_slice.displayed_dimensions,
            point=slice_request.world_slice.point,
            margin_negative=slice_request.world_slice.margin_negative,
            margin_positive=slice_request.world_slice.margin_positive,
        )


class MultiscaleImageDataStream(BaseImageDataStream):
    """Class for synchronous image data_stores streams."""

    data_store_id: str
    selectors: List[str]

    # this is used for a discriminated union
    stream_type: Literal["image_multiscale"] = "image_multiscale"

    def get_data_store_slice(
        self, slice_request: DataSliceRequest
    ) -> ImageDataStoreChunk:
        """Get slice object to get the requested world data slice from the data store.

        todo: handle mixed dimensions, etc.

        Parameters
        ----------
        slice_request : DataSliceRequest
            The requested data slice to generate the data store slice from.
        """
        raise NotImplementedError

        # return ImageDataStoreChunk(
        #     scene_id=slice_request.scene_id,
        #     visual_id=slice_request.visual_id,
        #     resolution_level=slice_request.resolution_level,
        #     array_coordinate_start=min_corner_array[[2, 1, 0]],
        #     array_coordinate_end=max_corner_array[[2, 1, 0]],
        #     texture_coordinate_start=min_corner_texture,
        # )
