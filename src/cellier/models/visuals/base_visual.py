"""Base classes for visuals and materials."""

from uuid import uuid4

from psygnal import EventedModel
from pydantic import Field


class BaseVisual(EventedModel):
    """Base model for all visuals."""

    name: str

    # store a UUID to identify this specific scene.
    id: str = Field(default_factory=lambda: uuid4().hex)


class BaseMaterial(EventedModel):
    """Base model for all materials."""

    pass
