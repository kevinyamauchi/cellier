"""RenderManager class contains all the rendering and visuals code."""

from dataclasses import dataclass
from functools import partial
from typing import Dict

import numpy as np
import pygfx
import pygfx as gfx
from psygnal import Signal
from pygfx.controllers import TrackballController
from pygfx.renderers import WgpuRenderer
from wgpu.gui import WgpuCanvasBase

from cellier.models.viewer import ViewerModel
from cellier.render.utils import construct_pygfx_object
from cellier.slicer.data_slice import (
    RenderedMeshDataSlice,
    RenderedPointsDataSlice,
    RenderedSliceData,
)

# class VisualKey(NamedTuple):
#     """The key to a visual stored in the RenderManager.
#
#     Attributes
#     ----------
#     scene_id : str
#         The uid of the scene model this visual belongs to.
#     visual_id : str
#         The uid of the visual model this pygfx belongs to.
#     """
#
#     scene_id: str
#     visual_id: str


@dataclass(frozen=True)
class CanvasRedrawRequest:
    """Data to request a redraw of all canvases in a scene."""

    scene_id: str


class RenderManagerEvents:
    """Events for the RenderManager class."""

    redraw_canvas: Signal = Signal(CanvasRedrawRequest)


class RenderManager:
    """Class to manage the rendering."""

    def __init__(self, viewer_model: ViewerModel, canvases: Dict[str, WgpuCanvasBase]):
        # add the events
        self.events = RenderManagerEvents()

        # make each scene
        renderers = {}
        cameras = {}
        scenes = {}
        visuals = {}
        controllers = {}
        for scene_model in viewer_model.scenes.scenes.values():
            # make a scene
            scene = gfx.Scene()

            # todo add lighting config
            scene.add(gfx.AmbientLight())

            # todo add scene decorations config
            axes = gfx.AxesHelper(size=5, thickness=8)
            scene.add(axes)

            # populate the scene
            for visual_model in scene_model.visuals:
                world_object = construct_pygfx_object(
                    visual_model=visual_model, data_manager=viewer_model.data
                )
                scene.add(world_object)
                visuals.update({visual_model.id: world_object})

                # add a bounding box
                # todo make configurable
                # box_world = gfx.BoxHelper(color="red")
                # box_world.set_transform_by_object(world_object)
                # scene.add(box_world)

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
                # camera.show_object(scene)
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

    def animate(self, scene_id: str, canvas_id: str) -> None:
        """Callback to render a given canvas."""
        renderer = self.renderers[canvas_id]
        renderer.render(self.scenes[scene_id], self.cameras[canvas_id])

    def _on_new_slice(
        self, slice_data: RenderedSliceData, redraw_canvas: bool = True
    ) -> None:
        """Callback to update objects when a new slice is received."""
        visual_object = self._visuals[slice_data.visual_id]

        if isinstance(slice_data, RenderedMeshDataSlice):
            new_geometry = gfx.Geometry(
                positions=slice_data.vertices, indices=slice_data.faces
            )
            visual_object.geometry = new_geometry

        elif isinstance(slice_data, RenderedPointsDataSlice):
            coordinates = slice_data.coordinates
            if coordinates.shape[1] == 2:
                # pygfx expects 3D points
                n_points = coordinates.shape[0]
                zeros_column = np.zeros((n_points, 1), dtype=np.float32)
                coordinates = np.column_stack((coordinates, zeros_column))

            if coordinates.shape[0] == 0:
                # coordinates must not be empty
                # todo do something smarter?
                coordinates = np.array([[0, 0, 0]], dtype=np.float32)
            new_geometry = gfx.Geometry(positions=coordinates)
            visual_object.geometry = new_geometry
        else:
            raise ValueError(f"Unrecognized slice data type: {slice_data}")

        if redraw_canvas:
            self.events.redraw_canvas.emit(
                CanvasRedrawRequest(scene_id=slice_data.scene_id)
            )
