"""Classes for lines DataStores."""

from typing import Literal

import numpy as np
from pydantic import ConfigDict, field_serializer, field_validator
from pydantic_core.core_schema import ValidationInfo

from cellier.models.data_stores.base_data_store import BaseDataStore
from cellier.types import (
    AxisAlignedDataRequest,
    AxisAlignedSelectedRegion,
    DataRequest,
    LinesDataResponse,
    PlaneDataRequest,
    PlaneSelectedRegion,
    SceneId,
    SelectedRegion,
    TilingMethod,
    VisualId,
)


class BaseLinesDataStore(BaseDataStore):
    """The base class for all lines data_stores.

    todo: properly set up. this shouldn't specify ndarrays.
    """

    coordinates: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("coordinates", mode="before")
    @classmethod
    def coerce_to_ndarray_float32(cls, v: str, info: ValidationInfo):
        """Coerce to a float32 numpy array."""
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.float32)
        return v.astype(np.float32)


class LinesMemoryStore(BaseLinesDataStore):
    """Point data_stores store for arrays stored in memory."""

    # this is used for a discriminated union
    store_type: Literal["lines_memory"] = "lines_memory"

    @field_serializer("coordinates")
    def serialize_ndarray(self, array: np.ndarray, _info) -> list:
        """Coerce numpy arrays into lists for serialization."""
        return array.tolist()

    def get_data_request(
        self,
        selected_region: SelectedRegion,
        tiling_method: TilingMethod,
        scene_id: SceneId,
        visual_id: VisualId,
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
            # determine the start of the chunk in the rendered coordinates
            displayed_dim_indices = selected_region.ordered_dims[
                -selected_region.n_displayed_dims :
            ]
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
            raise TypeError(f"Unknown region selection type: {type(selected_region)}")

    def get_data(
        self,
        request: DataRequest,
    ) -> LinesDataResponse:
        """Get the data for a given request.

        Parameters
        ----------
        request : DataRequest
            The request for the data.

        Returns
        -------
        PointsDataResponse
            The response containing the data.
        """
        if isinstance(request, AxisAlignedDataRequest):
            displayed_dimensions = list(
                request.ordered_dims[-request.n_displayed_dims :]
            )
            points_ndim = self.coordinates.shape[1]

            # get a mask for the not displayed dimensions
            not_displayed_mask = np.ones((points_ndim,), dtype=bool)
            not_displayed_mask[displayed_dimensions] = False

            # get the range to include
            n_not_displayed = not_displayed_mask.sum()
            not_displayed_low = np.zeros((n_not_displayed,))
            not_displayed_high = np.zeros((n_not_displayed,))
            not_displayed_selection = [
                selection
                for index, selection in enumerate(request.index_selection)
                if index not in displayed_dimensions
            ]
            for not_displayed_index, selection in enumerate(not_displayed_selection):
                if isinstance(selection, int):
                    not_displayed_low[not_displayed_index] = selection
                    not_displayed_high[not_displayed_index] = selection
                else:
                    not_displayed_low[not_displayed_index] = selection.start
                    if selection.stop > 0:
                        not_displayed_high[not_displayed_index] = selection.stop - 1
                    else:
                        not_displayed_high[not_displayed_index] = 0

            # find the coordinates inside the slice
            not_displayed_coordinates = self.coordinates[:, not_displayed_mask]

            inside_slice_mask = np.all(
                (not_displayed_coordinates >= not_displayed_low)
                & (not_displayed_coordinates <= not_displayed_high),
                axis=1,
            )

            in_slice_coordinates = np.atleast_2d(
                self.coordinates[inside_slice_mask, :][:, displayed_dimensions]
            )

            return LinesDataResponse(
                id=request.id,
                scene_id=request.scene_id,
                visual_id=request.visual_id,
                resolution_level=request.resolution_level,
                data=in_slice_coordinates,
            )

        elif isinstance(request, PlaneDataRequest):
            raise NotImplementedError("Plane samples are not implemented yet.")
        else:
            raise TypeError(f"Unknown request type: {type(request)}")
