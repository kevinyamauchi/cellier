"""Class to hold all of the data stores and streams."""

from typing import Dict

from psygnal import EventedModel

from cellier.models.data_stores.mesh import MeshMemoryStore
from cellier.models.data_streams.mesh import MeshSynchronousDataStream


class DataManager(EventedModel):
    """Class to model all data_stores in the viewer.

    todo: add discrimitive union
    """

    stores: Dict[str, MeshMemoryStore]
    streams: Dict[str, MeshSynchronousDataStream]
