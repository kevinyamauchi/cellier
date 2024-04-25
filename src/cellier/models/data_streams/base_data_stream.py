"""Base class for all data_stores streams."""

from uuid import uuid4

from psygnal import EventedModel
from pydantic import Field


class BaseDataStream(EventedModel):
    """Base class for all data_stores streams.

    Parameters
    ----------
    id : str
        The unique identifier for the data store.
        The default value is a UUID4 generated hex string.

    Attributes
    ----------
    id : str
        The unique identifier for this data stream instance.
    """

    # store a UUID to identify this specific scene.
    id: str = Field(default_factory=lambda: uuid4().hex)
