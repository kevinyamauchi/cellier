"""Models for all cameras."""

from typing import Literal, Union

from psygnal import EventedModel
from pydantic import Field
from typing_extensions import Annotated


class BaseCamera(EventedModel):
    """Base class for all camera models."""

    pass


class PerspectiveCamera(BaseCamera):
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

    # this is used for a discriminated union
    camera_type: Literal["perspective"] = "perspective"


class OrthographicCamera(BaseCamera):
    """Orthographic Camera model.

    See the PyGFX OrthographicCamera documentation
    for more details.

    This is a psygnal EventedModel.
    https://psygnal.readthedocs.io/en/latest/API/model/

    Parameters
    ----------
    ----------.
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

    width: float = 10
    height: float = 10
    zoom: float = 1
    near_clipping_plane: float = -500
    far_clipping_plane: float = 500

    # this is used for a discriminated union
    camera_type: Literal["orthographic"] = "orthographic"


CameraType = Annotated[
    Union[PerspectiveCamera, OrthographicCamera], Field(discriminator="camera_type")
]
