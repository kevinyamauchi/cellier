"""Models for mesh data_stores stores."""

from dataclasses import dataclass
from typing import Literal, Tuple, Union

import numpy as np
from pydantic import ConfigDict, ValidationInfo, field_serializer, field_validator

from cellier.models.data_stores.base_data_store import BaseDataStore, DataStoreSlice
from cellier.slicer.data_slice import RenderedMeshDataSlice


@dataclass(frozen=True)
class MeshDataStoreSlice(DataStoreSlice):
    """Class containing data to slice a mesh data store.

    Parameters
    ----------
    displayed_dimensions : Union[Tuple[int, int, int], Tuple[int, int]]
        The indices of the displayed dimensions.
        The indices are ordered in their display order.
    visual_id : str
        The UID of the corresponding visual.
    resolution_level : int
        The resolution level to render where 0 is the highest resolution
        and high numbers correspond with more down sampling.
    """

    displayed_dimensions: Union[Tuple[int, int, int], Tuple[int, int]]


class BaseMeshDataStore(BaseDataStore):
    """The base class for all mesh data_stores.

    todo: properly set up. this shouldn't specify ndarrays.
    """

    vertices: np.ndarray
    faces: np.ndarray
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("vertices", mode="before")
    @classmethod
    def coerce_to_ndarray_float32(cls, v: str, info: ValidationInfo):
        """Coerce to a float32 numpy array."""
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.float32)
        return v.astype(np.float32)

    @field_validator("faces", mode="before")
    @classmethod
    def coerce_to_ndarray_int32(cls, v, info: ValidationInfo):
        """Coerce to an int32 numpy array."""
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.int32)
        return v.astype(np.int32)

    @field_serializer("vertices", "faces")
    def serialize_ndarray(self, array: np.ndarray, _info) -> list:
        """Coerce numpy arrays into lists for serialization."""
        return array.tolist()


class MeshMemoryStore(BaseMeshDataStore):
    """Mesh data_stores store for arrays stored in memory."""

    # this is used for a discriminated union
    store_type: Literal["mesh_memory"] = "mesh_memory"

    def get_slice(self, slice_data: MeshDataStoreSlice) -> RenderedMeshDataSlice:
        """Get the data required to render a slice of the mesh."""
        displayed_dimensions = slice_data.displayed_dimensions
        vertices = self.vertices[:, displayed_dimensions]

        return RenderedMeshDataSlice(
            scene_id=slice_data.scene_id,
            visual_id=slice_data.visual_id,
            resolution_level=slice_data.resolution_level,
            vertices=vertices,
            faces=self.faces,
        )
