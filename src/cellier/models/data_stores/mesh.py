"""Models for mesh data_stores stores."""

import numpy as np
from pydantic import ConfigDict, ValidationInfo, field_serializer, field_validator

from cellier.models.data_stores.base_data_store import BaseDataStore


class BaseMeshDataStore(BaseDataStore):
    """The base class for all mesh data_stores streams.

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

    pass
