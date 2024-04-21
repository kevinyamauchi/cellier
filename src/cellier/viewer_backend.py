"""PyGFX viewer backend."""

from typing import TYPE_CHECKING

import pygfx as gfx

from cellier.qt_canvas import QtCanvas

if TYPE_CHECKING:
    from cellier.models.scene.cameras import PerspectiveCamera


class ViewerBackend:
    """PyGFX viewer backend."""

    def __init__(self, camera: gfx.PerspectiveCamera):
        self._scene = gfx.Scene()
        self.camera = camera

        self._canvas_widget = QtCanvas()

        # make the renderer
        self._renderer = gfx.WgpuRenderer(self._canvas_widget.wgpu_canvas)

        # Hook up the animate callback
        self._canvas_widget.wgpu_canvas.request_draw(self.animate)

    def animate(self):
        """Method to render the scene."""
        self._renderer.render(self._scene, self.camera)

    def add_layer(self, layer: gfx.WorldObject):
        """Add a layer to the viewer."""
        # needs to make the gfx world object
        self._scene.add(layer)

    @classmethod
    def from_models(cls, camera: "PerspectiveCamera"):
        """Instantiate the backend from the models."""
        gfx_camera = gfx.PerspectiveCamera(
            fov=camera.fov,
            width=camera.width,
            height=camera.height,
            zoom=camera.zoom,
            depth_range=(camera.near_clipping_plane, camera.far_clipping_plane),
        )
        return cls(camera=gfx_camera)
