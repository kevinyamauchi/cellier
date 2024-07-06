"""Classes for Image data stores."""

from dataclasses import dataclass
from typing import Literal, Tuple, Union

import numpy as np
from pydantic import ConfigDict, field_serializer, field_validator
from pydantic_core.core_schema import ValidationInfo

from cellier.models.data_stores.base_data_store import BaseDataStore, DataStoreSlice
from cellier.slicer.data_slice import RenderedImageDataSlice


@dataclass(frozen=True)
class ImageDataStoreSlice(DataStoreSlice):
    """Class containing data to slice a mesh data store.

    Parameters
    ----------
    displayed_dimensions : Union[Tuple[int, int, int], Tuple[int, int]]
        The indices of the displayed dimensions.
        The indices are ordered in their display order.
    scene_id : str
        The UID of the scene the visual is rendered in.
    visual_id : str
        The UID of the corresponding visual.
    resolution_level : int
        The resolution level to render where 0 is the highest resolution
        and high numbers correspond with more down sampling.
    point : tuple of floats
        Dims position in data coordinates for each dimension.
    margin_negative : tuple of floats
        Negative margin in data units of the slice for each dimension.
    margin_positive : tuple of floats
        Positive margin in data units of the slice for each dimension.
    """

    displayed_dimensions: Union[Tuple[int, int, int], Tuple[int, int]]
    point: Tuple[float, ...] = ()
    margin_negative: Tuple[float, ...] = ()
    margin_positive: Tuple[float, ...] = ()


class BaseImageDataStore(BaseDataStore):
    """The base class for all image data_stores."""

    name: str = "image_data_store"


class ImageMemoryStore(BaseImageDataStore):
    """Point data_stores store for arrays stored in memory."""

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

    def get_slice(self, slice_data: ImageDataStoreSlice) -> RenderedImageDataSlice:
        """Get the data required to render a slice of the mesh.

        todo: generalize to oblique slicing
        """
        displayed_dimensions = list(slice_data.displayed_dimensions)

        slice_objects = [
            int(point_value)
            if (dimension_index not in displayed_dimensions)
            else slice(None)
            for dimension_index, point_value in enumerate(slice_data.point)
        ]

        return RenderedImageDataSlice(
            scene_id=slice_data.scene_id,
            visual_id=slice_data.visual_id,
            resolution_level=slice_data.resolution_level,
            data=self.data[tuple(slice_objects)],
        )
