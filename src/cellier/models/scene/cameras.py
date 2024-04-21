"""Models for all cameras."""

from typing import Dict, Union

from psygnal import EventedModel


class PerspectiveCamera(EventedModel):
    """Perspective camera model.

    This is a psygnal EventedModel.
    https://psygnal.readthedocs.io/en/latest/API/model/

    Parameters
    ----------
    fov : float
        The field of view (in degrees), between 0-179.
    width : float
        The (minimum) width of the view-cube.
    height : float
        The (minimum) height of the view-cube.
    zoom : float
        The zoom factor.
    near_clipping_plane : float
        The location of the near-clipping plane.
    far_clipping_plane : float
        The location of the far-clipping plane.
    """

    fov: float = 50
    width: float = 10
    height: float = 10
    zoom: float = 1
    near_clipping_plane: float = -500
    far_clipping_plane: float = 500

    def to_dict(self) -> Dict[str, float]:
        """Get the state of the camera model."""
        return {
            "fov": self.fov,
            "width": self.width,
            "height": self.height,
            "zoom": self.zoom,
            "near_clipping_plane": self.near_clipping_plane,
            "far_clipping_plane": self.far_clipping_plane,
        }


BaseCamera = Union[PerspectiveCamera]
