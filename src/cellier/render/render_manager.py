"""RenderManager class contains all the rendering and visuals code."""

from functools import partial
from typing import Dict, NamedTuple

import pygfx
import pygfx as gfx
from pygfx.controllers import TrackballController
from pygfx.renderers import WgpuRenderer
from wgpu.gui import WgpuCanvasBase

from cellier.models.viewer import ViewerModel
from cellier.render.utils import construct_pygfx_object


class VisualKey(NamedTuple):
    """The key to a visual stored in the RenderManager.

    Attributes
    ----------
    scene_id : str
        The uid of the scene model this visual belongs to.
    visual_id : str
        The uid of the visual model this pygfx belongs to.
    """

    scene_id: str
    visual_id: str


class RenderManager:
    """Class to manage the rendering."""

    def __init__(self, viewer_model: ViewerModel, canvases: Dict[str, WgpuCanvasBase]):
        # make each scene
        renderers = {}
        cameras = {}
        scenes = {}
        visuals = {}
        controllers = {}
        for scene_model in viewer_model.scenes.scenes:
            # make a scene
            scene = gfx.Scene()
            scene.add(gfx.AmbientLight())
            import numpy as np

            positions = np.random.normal(15, 3, (100, 3)).astype(np.float32)
            sizes = np.random.rand(100).astype(np.float32) * 50
            colors = np.random.rand(100, 4).astype(np.float32)
            geometry = gfx.Geometry(positions=positions, sizes=sizes, colors=colors)

            material = gfx.PointsMaterial(color_mode="vertex", size_mode="vertex")
            points = gfx.Points(geometry, material)
            scene.add(points)

            # populate the scene
            for visual_model in scene_model.visuals:
                key = VisualKey(scene_id=scene_model.id, visual_id=visual_model.id)
                world_object = construct_pygfx_object(visual_model=visual_model)
                scene.add(world_object)
                visuals.update({key: world_object})

            # store the scene
            scene_id = scene_model.id
            scenes.update({scene_id: scene})

            for canvas_model in scene_model.canvases:
                # make a renderer for each canvas
                canvas_id = canvas_model.id
                canvas = canvases[canvas_id]
                renderer = WgpuRenderer(canvas)
                renderers.update({canvas_id: renderer})

                # make a camera for each canvas
                # camera = construct_pygfx_camera_from_model(
                #         camera_model=canvas_model.camera
                # )
                camera = gfx.OrthographicCamera(110, 110)
                camera.show_object(scene)
                cameras.update({canvas_id: camera})

                # make the camera controller
                controller = TrackballController(
                    camera=camera, register_events=renderer
                )
                controllers.update({canvas_id: controller})

                # connect a callback for the renderer
                # todo should this be outside the renderer?
                render_func = partial(
                    self.animate, scene_id=scene_id, canvas_id=canvas_id
                )
                canvas.request_draw(render_func)

        # store the values
        self._scenes = scenes
        self._visuals = visuals
        self._renderers = renderers
        self._cameras = cameras
        self._controllers = controllers

    @property
    def renderers(self) -> Dict[str, WgpuRenderer]:
        """Dictionary of pygfx renderers.

        The key is the id property of the Canvas model the renderer
        belongs to.
        """
        return self._renderers

    @property
    def cameras(self) -> Dict[str, pygfx.Camera]:
        """Dictionary of pygfx Cameras.

        The key is the id property of the Canvas model the Camera
        belongs to.
        """
        return self._cameras

    @property
    def scenes(self) -> Dict[str, gfx.Scene]:
        """Dictionary of pygfx Scenes.

        The key is the id of the Scene model the pygfx Scene belongs to.
        """
        return self._scenes

    def animate(self, scene_id: str, canvas_id: str):
        """Callback to render a given canvas."""
        renderer = self.renderers[canvas_id]
        renderer.render(self.scenes[scene_id], self.cameras[canvas_id])
