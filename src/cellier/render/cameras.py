"""Functions to construct PyGFX cameras."""

from pygfx import OrbitController as GFXOrbitController
from pygfx import PanZoomController as GFXPanZoomController
from pygfx import PerspectiveCamera as GFXPerspectiveCamera
from pygfx import TrackballController as GFXTrackballController

from cellier.models.scene import (
    OrbitCameraController,
    OrthographicCamera,
    PanZoomCameraController,
    PerspectiveCamera,
    TrackballCameraController,
)


def construct_pygfx_camera_from_model(
    camera_model: PerspectiveCamera,
) -> tuple[
    GFXPerspectiveCamera,
    GFXOrbitController | GFXTrackballController | GFXPanZoomController,
]:
    """Make a pygfx perspective camera.

    todo: make general constructor for other cameras

    Parameters
    ----------
    camera_model : PerspectiveCamera
        The cellier PerspectiveCamera model to construct
        the PyGFX camera from.
    """
    if isinstance(camera_model, PerspectiveCamera):
        camera = GFXPerspectiveCamera(
            fov=camera_model.fov,
            width=camera_model.width,
            height=camera_model.height,
            zoom=camera_model.zoom,
        )
    elif isinstance(camera_model, OrthographicCamera):
        camera = GFXPerspectiveCamera(
            fov=0,
            width=camera_model.width,
            height=camera_model.height,
            zoom=camera_model.zoom,
        )
    else:
        raise ValueError(f"Unsupported camera model type: {camera_model}")

    if isinstance(camera_model.controller, PanZoomCameraController):
        controller = GFXPanZoomController(
            camera=camera,
            enabled=camera_model.controller.enabled,
        )
    elif isinstance(camera_model.controller, OrbitCameraController):
        controller = GFXOrbitController(
            camera=camera,
            enabled=camera_model.controller.enabled,
        )
    elif isinstance(camera_model.controller, TrackballCameraController):
        controller = GFXTrackballController(
            camera=camera,
            enabled=camera_model.controller.enabled,
        )
    else:
        raise ValueError(
            f"Unsupported camera controller type: {camera_model.controller}"
        )

    return camera, controller
