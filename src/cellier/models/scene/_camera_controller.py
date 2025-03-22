from typing import Literal, Union
from uuid import uuid4

from psygnal import EventedModel
from pydantic import Field
from typing_extensions import Annotated


class BaseCameraController(EventedModel):
    """Base class for all camera controllers.

    This is a psygnal EventedModel.
    https://psygnal.readthedocs.io/en/latest/API/model/

    Parameters
    ----------
    id : str
        The unique identifier for this camera controller.
        This is populated automatically by the model using uuid4.
    enabled : bool
        If set to True, the controller is active.
    """

    enabled: bool = True

    # store a UUID to identify this specific controller.
    id: str = Field(default_factory=lambda: uuid4().hex)


class PanZoomCameraController(BaseCameraController):
    """Pan and zoom controller for the camera.

    This is a psygnal EventedModel.
    https://psygnal.readthedocs.io/en/latest/API/model/

    Parameters
    ----------
    id : str
        The unique identifier for this camera controller.
        This is populated automatically by the model using uuid4.
    enabled : bool
        If set to True, the controller is active.
    """

    # this is used for a discriminated union
    controller_type: Literal["pan_zoom"] = "pan_zoom"


class TrackballCameraController(BaseCameraController):
    """Trackball camera controller.

    This controller allows the camera to freely rotate around center point.

    This is a psygnal EventedModel.
    https://psygnal.readthedocs.io/en/latest/API/model/

    Parameters
    ----------
    id : str
        The unique identifier for this camera controller.
        This is populated automatically by the model using uuid4.
    enabled : bool
        If set to True, the controller is active.
    """

    # this is used for a discriminated union
    controller_type: Literal["trackball"] = "trackball"


class OrbitCameraController(BaseCameraController):
    """Orbit camera controller.

    This controller allows the camera to rotate around a target point
    while maintaining an up direction.

    This is a psygnal EventedModel.
    https://psygnal.readthedocs.io/en/latest/API/model/

    Parameters
    ----------
    id : str
        The unique identifier for this camera controller.
        This is populated automatically by the model using uuid4.
    enabled : bool
        If set to True, the controller is active.
    """

    # this is used for a discriminated union
    controller_type: Literal["orbit"] = "orbit"


CameraControllerType = Annotated[
    Union[PanZoomCameraController, TrackballCameraController, OrbitCameraController],
    Field(discriminator="controller_type"),
]
