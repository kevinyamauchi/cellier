"""Class to hold all of the data stores and streams."""

import uuid
from typing import Dict, Union

from psygnal import EventedModel
from pydantic import Field
from typing_extensions import Annotated

from cellier.models.data_stores.image import (
    ImageMemoryStore,
    MockLatentImageStore,
    MultiScaleImageZarrStore,
)
from cellier.models.data_stores.mesh import MeshMemoryStore
from cellier.models.data_stores.points import PointsMemoryStore
from cellier.models.data_streams.image import (
    ImageSynchronousDataStream,
    MultiscaleImageDataStream,
)
from cellier.models.data_streams.mesh import MeshSynchronousDataStream
from cellier.models.data_streams.points import PointsSynchronousDataStream

# types for discrimitive unions
DataStoreType = Annotated[
    Union[
        ImageMemoryStore,
        MultiScaleImageZarrStore,
        MockLatentImageStore,
        MeshMemoryStore,
        PointsMemoryStore,
    ],
    Field(discriminator="store_type"),
]
DataStreamType = Annotated[
    Union[
        ImageSynchronousDataStream,
        MultiscaleImageDataStream,
        MeshSynchronousDataStream,
        PointsSynchronousDataStream,
    ],
    Field(discriminator="stream_type"),
]


class DataManager(EventedModel):
    """Class to model all data_stores in the viewer.

    todo: add discrimitive union

    Attributes
    ----------
    stores : Dict[str, DataStoreType]
        The data stores in the viewer.
        The key to the store is the data store id.

    """

    stores: Dict[str, DataStoreType]

    def add_data_store(self, data_store: DataStoreType):
        """Add a data store to the viewer.

        Parameters
        ----------
        data_store : DataStoreType
            The data store to add to the viewer.

        """
        self.stores[data_store.id] = data_store

        # emit event to signal that the data has been updated
        self.events.stores.emit()
