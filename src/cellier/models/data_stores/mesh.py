"""Models for mesh data stores."""

import numpy as np
from pydantic import ConfigDict

from cellier.models.data_stores.base_data_store import BaseDataStore


class BaseMeshDataStore(BaseDataStore):
    """The base class for all mesh data streams.

    todo: properly set up. this shouldn't specify ndarrays.
    """

    vertices: np.ndarray
    faces: np.ndarray
    model_config = ConfigDict(arbitrary_types_allowed=True)


class MeshMemoryStore(BaseMeshDataStore):
    """Mesh data store for arrays stored in memory."""

    pass
