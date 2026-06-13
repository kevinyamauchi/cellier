"""Base class for data stores."""

import uuid
from typing import Annotated
from uuid import uuid4

from psygnal import EventedModel
from pydantic import UUID4, AfterValidator, Field


class BaseDataStore(EventedModel):
    """The base class for all DataStores.

    Parameters
    ----------
    id : UUID4
        The unique identifier for the data store.
        The default value is a UUID4 generated hex string.
    name : str
        The name of the data store.

    Attributes
    ----------
    id : str
        The unique identifier for the data store.
    """

    # store a UUID to identify this specific scene.
    id: UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))] = (
        Field(frozen=True, default_factory=lambda: uuid4())
    )
    name: str = "data store"
