"""Data stores for Image data.

These can be used for both intensity and label images.
"""

from typing import Literal

import numpy as np
from pydantic import ConfigDict, field_serializer, field_validator
from pydantic_core.core_schema import ValidationInfo

from cellier.models.data_stores.base_data_store import BaseDataStore
from cellier.slicer import DataSliceRequest, RenderedImageDataSlice
from cellier.types import (
    AxisAlignedDataRequest,
    AxisAlignedSample,
    DataRequest,
    ImageDataResponse,
    SceneId,
    TilingMethod,
    VisualId,
)


class BaseImageDataStore(BaseDataStore):
    """The base class for all image data_stores."""

    name: str = "image_data_store"


class ImageMemoryStore(BaseImageDataStore):
    """Image data store for arrays stored in memory.

    Parameters
    ----------
    name : str
        The name of the data store.
    data : np.ndarray
        The data to be stored.
    """

    data: np.ndarray

    # this is used for a discriminated union
    store_type: Literal["image_memory"] = "image_memory"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("data", mode="before")
    @classmethod
    def coerce_to_ndarray_float32(cls, v: str, info: ValidationInfo):
        """Coerce to a float32 numpy array."""
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.float32)
        return v.astype(np.float32)

    @field_serializer("data")
    def serialize_ndarray(self, array: np.ndarray, _info) -> list:
        """Coerce numpy arrays into lists for serialization."""
        return array.tolist()

    def get_data_request(
        self,
        selected_region: AxisAlignedSample,
        visual_id: VisualId,
        scene_id: SceneId,
    ) -> list[DataRequest]:
        """Get the data requests for a given selected region."""
        if selected_region.tiling_method != TilingMethod.NONE:
            raise NotImplementedError(
                "Tiling is not implemented for the ImageMemoryStore."
            )
        if isinstance(selected_region, AxisAlignedSample):
            displayed_dim_indices = selected_region.ordered_dims[
                -selected_region.n_displayed_dims :
            ]

            # determine the start of the chunk in the rendered coordinates
            min_corner_rendered = ()
            for axis_index in displayed_dim_indices:
                selection = selected_region.index_selection[axis_index]
                if isinstance(selection, int):
                    min_corner_rendered += (selection,)
                else:
                    if selection.start is None:
                        min_corner_rendered += (0,)
                    else:
                        min_corner_rendered += (selection.start,)
            return [
                AxisAlignedDataRequest(
                    visual_id=visual_id,
                    scene_id=scene_id,
                    min_corner_rendered=min_corner_rendered,
                    ordered_dims=selected_region.ordered_dims,
                    n_displayed_dims=selected_region.n_displayed_dims,
                    resolution_level=0,
                    index_selection=selected_region.index_selection,
                )
            ]

        else:
            raise TypeError(f"Unexpected selected region type: {type(selected_region)}")

    def get_data(self, request: DataRequest) -> ImageDataResponse:
        """Get the data for a given request.

        Parameters
        ----------
        request : DataRequest
            The request for the data.

        Returns
        -------
        ImageDataResponse
            The response containing the data.
        """
        return ImageDataResponse(
            id=request.id,
            scene_id=request.scene_id,
            visual_id=request.visual_id,
            resolution_level=request.resolution_level,
            data=self.data[request.index_selection],
            min_corner_rendered=request.min_corner_rendered,
        )

    def get_slice(self, slice_data: DataSliceRequest) -> RenderedImageDataSlice:
        """Get the data required to render a slice of the mesh.

        todo: generalize to oblique slicing
        """
        displayed_dimensions = list(slice_data.world_slice.displayed_dimensions)

        slice_objects = [
            int(point_value)
            if (dimension_index not in displayed_dimensions)
            else slice(None)
            for dimension_index, point_value in enumerate(slice_data.world_slice.point)
        ]

        return RenderedImageDataSlice(
            scene_id=slice_data.scene_id,
            visual_id=slice_data.visual_id,
            resolution_level=slice_data.resolution_level,
            data=self.data[tuple(slice_objects)],
            texture_start_index=(0, 0, 0),
        )
