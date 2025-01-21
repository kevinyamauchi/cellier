"""Models for all cameras."""

from typing import Literal, Union

import numpy as np
from psygnal import EventedModel
from pydantic import ConfigDict, Field, field_serializer, field_validator
from pydantic_core.core_schema import ValidationInfo
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
    position: np.ndarray = np.array([0, 0, 0])
    rotation: np.ndarray = np.array([0, 0, 0, 0])
    up_direction: np.ndarray = np.array([0, 0, 0])
    frustum: np.ndarray = np.zeros((2, 4, 3))

    # this is used for a discriminated union
    camera_type: Literal["perspective"] = "perspective"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("position", "rotation", "up_direction", "frustum", mode="before")
    @classmethod
    def coerce_to_ndarray_float32(cls, v: str, info: ValidationInfo):
        """Coerce to a float32 numpy array."""
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.float32)
        return v.astype(np.float32)

    @field_serializer("position", "rotation", "up_direction", "frustum")
    def serialize_ndarray(self, array: np.ndarray, _info) -> list:
        """Coerce numpy arrays into lists for serialization."""
        return array.tolist()


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
