"""Models for the scene canvas."""

from uuid import uuid4

from psygnal import EventedModel
from pydantic import Field

from cellier.models.scene.cameras import BaseCamera


class Canvas(EventedModel):
    """Model for the scene canvas.

    Parameters
    ----------
    camera : BaseCamera
    """

    camera: BaseCamera

    # store a UUID to identify this specific scene.
    id: str = Field(default_factory=lambda: uuid4().hex)
