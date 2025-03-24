"""Models for all cameras."""

from dataclasses import dataclass
from typing import Any, Literal, Union
from uuid import uuid4

import numpy as np
from psygnal import EmissionInfo, EventedModel
from pydantic import ConfigDict, Field, field_serializer, field_validator
from pydantic_core.core_schema import ValidationInfo
from typing_extensions import Annotated

from cellier.models.scene._camera_controller import (
    CameraControllerState,
    CameraControllerType,
)
from cellier.types import CameraId


@dataclass(frozen=True)
class CameraState:
    """The (frozen) state of the camera."""

    id: CameraId
    controller: CameraControllerState
    fov: float
    width: float
    height: float
    zoom: float
    near_clipping_plane: float
    far_clipping_plane: float
    position: np.ndarray
    rotation: np.ndarray
    up_direction: np.ndarray
    frustum: np.ndarray


class BaseCamera(EventedModel):
    """Base class for all camera models.

    Parameters
    ----------
    id : str
        The unique identifier for this camera.
        This is populated automatically by the model using uuid4.
    controller : CameraControllerType
        The camera controller for this camera.
        The camera controller is responsible for updating the camera
        in response to mouse events.
    """

    controller: CameraControllerType

    # store a UUID to identify this specific scene.
    id: str = Field(default_factory=lambda: uuid4().hex)

    def model_post_init(self, __context: Any) -> None:
        """Set make the controller event fire when the controller model is updated."""
        self.controller.events.all.connect(self._on_controller_updated)

    def update_state(self, new_state):
        """Update the state of the visual.

        This is often used as a callback for when
        the visual controls update.
        """
        # remove the id field from the new state if present
        new_state.pop("id", None)

        # update the visual with the new state
        self.update(new_state)

    def _on_controller_updated(self, event: EmissionInfo):
        """Emit the controller() signal when the controller is updated."""
        property_name = event.signal.name
        property_value = event.args[0]
        self.events.controller.emit({"controller": {property_name: property_value}})


class PerspectiveCamera(BaseCamera):
    """Perspective camera model.

    This is a psygnal EventedModel.
    https://psygnal.readthedocs.io/en/latest/API/model/

    Parameters
    ----------
    fov : float
        The field of view (in degrees), between 0-179.
        To use orthographic projection, set the fov to 0.
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
    id : str
        The unique identifier for this camera.
        This is populated automatically by the model using uuid4.
    controller : CameraControllerType
        The camera controller for this camera.
        The camera controller is responsible for updating the camera
        in response to mouse events.
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

    def to_state(self) -> CameraState:
        """Convert the camera model to a frozen CameraState object.

        Returns
        -------
        CameraState
            The state of the camera.
        """
        return CameraState(
            id=self.id,
            controller=self.controller.to_state(),
            fov=self.fov,
            width=self.width,
            height=self.height,
            zoom=self.zoom,
            near_clipping_plane=self.near_clipping_plane,
            far_clipping_plane=self.far_clipping_plane,
            position=self.position,
            rotation=self.rotation,
            up_direction=self.up_direction,
            frustum=self.frustum,
        )


class OrthographicCamera(BaseCamera):
    """Orthographic Camera model.

    See the PyGFX OrthographicCamera documentation
    for more details.

    This is a psygnal EventedModel.
    https://psygnal.readthedocs.io/en/latest/API/model/

    Parameters
    ----------
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
    id : str
        The unique identifier for this camera.
        This is populated automatically by the model using uuid4.
    controller : CameraControllerType
        The camera controller for this camera.
        The camera controller is responsible for updating the camera
        in response to mouse events.
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
