"""Models for the scene canvas."""

from uuid import uuid4

from psygnal import EventedModel
from pydantic import Field

from cellier.models.scene.cameras import CameraType


class Canvas(EventedModel):
    """Model for the scene canvas.

    Parameters
    ----------
    camera : BaseCamera
    """

    camera: CameraType

    # store a UUID to identify this specific scene.
    id: str = Field(default_factory=lambda: uuid4().hex)
