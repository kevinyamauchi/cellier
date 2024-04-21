"""Base classes for visuals and materials."""

from psygnal import EventedModel


class BaseVisual(EventedModel):
    """Base model for all visuals."""

    name: str


class BaseMaterial(EventedModel):
    """Base model for all materials."""

    pass
