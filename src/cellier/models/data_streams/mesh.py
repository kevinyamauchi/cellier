"""Models for mesh data_stores streams."""

from typing import List

from cellier.models.data_streams.base_data_stream import BaseDataStream


class BaseMeshDataStream(BaseDataStream):
    """Base class for all mesh data_stores streams."""

    pass


class MeshSynchronousDataStream(BaseMeshDataStream):
    """Class for synchronous mesh data_stores streams."""

    data_store_id: str
    selectors: List[str]
