"""Classes for Point DataStores."""

from typing import Literal

import numpy as np
from pydantic import ConfigDict, field_serializer, field_validator
from pydantic_core.core_schema import ValidationInfo

from cellier.models.data_stores.base_data_store import BaseDataStore
from cellier.slicer.data_slice import DataSliceRequest, RenderedPointsDataSlice


class BasePointsDataStore(BaseDataStore):
    """The base class for all point data_stores.

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


class PointsMemoryStore(BasePointsDataStore):
    """Point data_stores store for arrays stored in memory."""

    # this is used for a discriminated union
    store_type: Literal["points_memory"] = "points_memory"

    @field_serializer("coordinates")
    def serialize_ndarray(self, array: np.ndarray, _info) -> list:
        """Coerce numpy arrays into lists for serialization."""
        return array.tolist()

    def get_slice(self, slice_data: DataSliceRequest) -> RenderedPointsDataSlice:
        """Get the data required to render a slice of the mesh.

        todo: generalize to oblique slicing
        """
        displayed_dimensions = list(slice_data.world_slice.displayed_dimensions)
        points_ndim = self.coordinates.shape[1]

        # get a mask for the not displayed dimensions
        not_displayed_mask = np.ones((points_ndim,), dtype=bool)
        not_displayed_mask[displayed_dimensions] = False

        # get the range to include
        point = np.asarray(slice_data.world_slice.point)
        margin_negative = np.asarray(slice_data.world_slice.margin_negative)
        margin_positive = np.asarray(slice_data.world_slice.margin_positive)
        low = point - margin_negative
        high = point + margin_positive

        # get the components of the range from the not displayed dimensions
        not_displayed_low = low[not_displayed_mask]
        not_displayed_high = high[not_displayed_mask]

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

        return RenderedPointsDataSlice(
            scene_id=slice_data.scene_id,
            visual_id=slice_data.visual_id,
            resolution_level=slice_data.resolution_level,
            coordinates=in_slice_coordinates,
        )
