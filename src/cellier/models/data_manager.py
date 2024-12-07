"""Class to hold all of the data stores and streams."""

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
    """

    stores: Dict[str, DataStoreType]
    streams: Dict[str, DataStreamType]
