"""Models for the scene canvas."""

from psygnal import EventedModel

from cellier.models.scene.cameras import BaseCamera


class Canvas(EventedModel):
    """Model for the scene canvas.

    Parameters
    ----------
    camera : BaseCamera
    """

    camera: BaseCamera
