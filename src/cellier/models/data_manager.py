"""Class to hold all of the data stores and streams."""

from typing import Dict, Union

from psygnal import EventedModel
from pydantic import Field
from typing_extensions import Annotated

from cellier.models.data_stores import (
    ImageMemoryStore,
    LinesMemoryStore,
    PointsMemoryStore,
)

# types for discrimitive unions
DataStoreType = Annotated[
    Union[PointsMemoryStore, LinesMemoryStore, ImageMemoryStore],
    Field(discriminator="store_type"),
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
