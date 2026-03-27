"""Discriminated union of all data store types for cellier v2."""

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from cellier.v2.data.image._image_memory_store import ImageMemoryStore
from cellier.v2.data.image._zarr_multiscale_store import MultiscaleZarrDataStore

DataStoreType = Annotated[
    Union[MultiscaleZarrDataStore, ImageMemoryStore],
    Field(discriminator="store_type"),
]
