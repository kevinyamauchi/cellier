"""Functions to construct PyGFX cameras."""

from pygfx import PerspectiveCamera as GFXPerspectiveCamera

from cellier.models.scene.cameras import PerspectiveCamera


def construct_pygfx_camera_from_model(
    camera_model: PerspectiveCamera,
) -> GFXPerspectiveCamera:
    """Make a pygfx perspective camera.

    todo: make general constructor for other cameras

    Parameters
    ----------
    camera_model : PerspectiveCamera
        The cellier PerspectiveCamera model to construct
        the PyGFX camera from.
    """
    return GFXPerspectiveCamera(
        fov=camera_model.fov,
        width=camera_model.width,
        height=camera_model.height,
        zoom=camera_model.zoom,
    )
