"""Discriminated union of all data store types for cellier v2."""

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from cellier.v2.data.image._image_memory_store import ImageMemoryStore
from cellier.v2.data.image._ome_zarr_image_store import OMEZarrImageDataStore
from cellier.v2.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.v2.data.label._label_memory_store import LabelMemoryStore
from cellier.v2.data.lines._lines_memory_store import LinesMemoryStore
from cellier.v2.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.v2.data.points._points_memory_store import PointsMemoryStore

DataStoreType = Annotated[
    Union[
        MultiscaleZarrDataStore,
        ImageMemoryStore,
        OMEZarrImageDataStore,
        LabelMemoryStore,
        PointsMemoryStore,
        LinesMemoryStore,
        MeshMemoryStore,
    ],
    Field(discriminator="store_type"),
]
