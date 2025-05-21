"""Data stores for Image data.

These can be used for both intensity and label images.
"""

from typing import Literal

import numpy as np
from pydantic import ConfigDict, field_serializer, field_validator
from pydantic_core.core_schema import ValidationInfo
from pylinalg import vec_transform
from scipy.ndimage import map_coordinates

from cellier.models.data_stores.base_data_store import BaseDataStore
from cellier.types import (
    AxisAlignedDataRequest,
    AxisAlignedSelectedRegion,
    DataRequest,
    ImageDataResponse,
    PlaneDataRequest,
    PlaneSelectedRegion,
    SceneId,
    SelectedRegion,
    TilingMethod,
    VisualId,
)
from cellier.util.geometry import generate_2d_grid


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
        selected_region: SelectedRegion,
        tiling_method: TilingMethod,
        visual_id: VisualId,
        scene_id: SceneId,
    ) -> list[DataRequest]:
        """Get the data requests for a given selected region.

        Parameters
        ----------
        selected_region : SelectedRegion
            The selected region to get data for.
        tiling_method : TilingMethod
            The method for computing how to chunk the request.
        visual_id : VisualId
            The unique identifier for the visual.
        scene_id : SceneId
            The unique identifier for the scene.
        """
        if tiling_method != TilingMethod.NONE:
            raise NotImplementedError(
                "Tiling is not implemented for the ImageMemoryStore."
            )
        if isinstance(selected_region, AxisAlignedSelectedRegion):
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

        elif isinstance(selected_region, PlaneSelectedRegion):
            # this is a 2D slice of the data
            return [
                PlaneDataRequest(
                    visual_id=visual_id,
                    scene_id=scene_id,
                    min_corner_rendered=(0, 0),
                    ordered_dims=selected_region.ordered_dims,
                    n_displayed_dims=selected_region.n_displayed_dims,
                    index_selection=selected_region.index_selection,
                    resolution_level=0,
                    point=selected_region.point,
                    plane_transform=selected_region.plane_transform,
                    extents=selected_region.extents,
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
        transposed_data = np.transpose(self.data, request.ordered_dims)
        transposed_selection = tuple(
            [request.index_selection[axis_index] for axis_index in request.ordered_dims]
        )

        if isinstance(request, AxisAlignedDataRequest):
            return ImageDataResponse(
                id=request.id,
                scene_id=request.scene_id,
                visual_id=request.visual_id,
                resolution_level=request.resolution_level,
                data=transposed_data[transposed_selection],
                min_corner_rendered=request.min_corner_rendered,
            )
        elif isinstance(request, PlaneDataRequest):
            # get the requested plane of data
            grid_shape = request.extents
            sampling_grid = generate_2d_grid(grid_shape=grid_shape, grid_spacing=(1, 1))
            grid_coords = sampling_grid.reshape(-1, 3)

            # apply the transform to the grid
            grid_transform = np.eye(4, dtype=np.float32)
            grid_transform[:3, :3] = request.plane_transform
            grid_transform[:3, 3] = request.point
            transformed_grid = vec_transform(grid_coords, grid_transform)

            # get the transformed grid in the nD coordinate system
            n_points = transformed_grid.shape[0]
            n_dims = len(request.ordered_dims)
            dims_index = np.zeros((1, n_dims))
            for ordered_dim_index, and_dim_index in enumerate(
                request.ordered_dims[: -request.n_displayed_dims]
            ):
                dims_index[0, and_dim_index] = request.index_selection[
                    ordered_dim_index
                ]
            transformed_grid_and = np.repeat(dims_index, repeats=n_points, axis=0)

            for displayed_dim_index in range(request.n_displayed_dims):
                dim_index_and = request.ordered_dims[
                    -(request.n_displayed_dims - displayed_dim_index)
                ]
                transformed_grid_and[:, dim_index_and] = transformed_grid[
                    :, displayed_dim_index
                ]

            # get the data
            sampled_volume = map_coordinates(
                self.data,
                transformed_grid_and.reshape(-1, n_dims).T,
                order=0,
                cval=0,
            )

            return ImageDataResponse(
                id=request.id,
                scene_id=request.scene_id,
                visual_id=request.visual_id,
                resolution_level=request.resolution_level,
                data=sampled_volume.reshape(grid_shape),
                min_corner_rendered=request.min_corner_rendered,
            )

        else:
            raise TypeError(f"Unexpected request type: {type(request)}")
