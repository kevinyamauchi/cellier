"""Base classes for nodes and materials."""

from uuid import uuid4

from psygnal import EventedModel
from pydantic import Field


class BaseNode(EventedModel):
    """Base model for all nodes."""

    name: str

    # store a UUID to identify this specific scene.
    id: str = Field(default_factory=lambda: uuid4().hex)


class BaseMaterial(EventedModel):
    """Base model for all materials."""

    pass
