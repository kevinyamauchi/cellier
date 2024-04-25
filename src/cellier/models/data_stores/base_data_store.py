"""Base Data Store classes."""

from uuid import uuid4

from psygnal import EventedModel
from pydantic import Field


class BaseDataStore(EventedModel):
    """The base class for all DataStores.

    Parameters
    ----------
    id : str
        The unique identifier for the data store.
        The default value is a UUID4 generated hex string.

    Attributes
    ----------
    id : str
        The unique identifier for the data store.
    """

    # store a UUID to identify this specific scene.
    id: str = Field(default_factory=lambda: uuid4().hex)
