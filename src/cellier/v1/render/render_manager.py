"""RenderManager class contains all the rendering and nodes code."""

from dataclasses import dataclass
from functools import partial
from typing import Dict

import numpy as np
import pygfx
import pygfx as gfx
from psygnal import Signal
from pygfx.controllers import TrackballController
from pygfx.renderers import WgpuRenderer
from superqt import ensure_main_thread
from wgpu.gui import WgpuCanvasBase

from cellier.models.viewer import ViewerModel
from cellier.render.cameras import construct_pygfx_camera_from_model
from cellier.render.utils import construct_pygfx_object
from cellier.slicer.data_slice import (
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


@dataclass(frozen=True)
class CameraState:
    """The current state of a camera.

    This should be the uniform across renderer implementations. Thus, should
    probably be moved outside the pygfx implementation.
    """

    scene_id: str
    canvas_id: str
    camera_id: str
    position: np.ndarray
    rotation: np.ndarray
    fov: float
    width: float
    height: float
    zoom: float
    up_direction: np.ndarray
    frustum: np.ndarray


class RenderManagerEvents:
    """Events for the RenderManager class."""

    redraw_canvas: Signal = Signal(CanvasRedrawRequest)
    camera_updated: Signal = Signal(CameraState)


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
                    node_model=visual_model
                )
                scene.add(world_object.node)
                visuals.update({visual_model.id: world_object})

                # add a bounding box
                # todo make configurable
                # box_world = gfx.BoxHelper(color="red")
                # box_world.set_transform_by_object(world_object)
                # scene.add(box_world)

            # store the scene
            scene_id = scene_model.id
            scenes.update({scene_id: scene})

            for canvas_id, canvas_model in scene_model.canvases.items():
                # make a renderer for each canvas
                canvas = canvases[canvas_id]
                renderer = WgpuRenderer(canvas)
                renderers.update({canvas_id: renderer})

                # make a camera for each canvas
                camera = construct_pygfx_camera_from_model(
                    camera_model=canvas_model.camera
                )
                # camera = gfx.PerspectiveCamera(width=110, height=110)
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

        self.render_calls = 0

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

    @property
    def visuals(self) -> Dict[str, gfx.WorldObject]:
        """The visuals in the RenderManager."""
        return self._visuals

    def add_visual(self, visual_model, scene_id: str):
        """Add a visual to a scene."""

        # get the scene node
        scene = self._scenes[scene_id]

        # get the visual object
        world_object = construct_pygfx_object(
            node_model=visual_model
        )

        # add the visual to the scene
        scene.add(world_object.node)

        self._visuals.update({visual_model.id: world_object})

    def animate(self, scene_id: str, canvas_id: str) -> None:
        """Callback to render a given canvas."""
        renderer = self.renderers[canvas_id]
        renderer.render(self.scenes[scene_id], self.cameras[canvas_id])

        # Send event to update the cameras
        camera = self.cameras[canvas_id]
        camera_state = camera.get_state()

        self.events.camera_updated.emit(
            CameraState(
                scene_id=scene_id,
                canvas_id=canvas_id,
                camera_id=camera.id,
                position=camera_state["position"],
                rotation=camera_state["rotation"],
                fov=camera_state["fov"],
                width=camera_state["width"],
                height=camera_state["height"],
                zoom=camera_state["zoom"],
                up_direction=camera_state["reference_up"],
                frustum=camera.frustum,
            )
        )

        self.render_calls += 1
        print(f"render: {self.render_calls}")

    @ensure_main_thread
    def _on_new_slice(
        self, slice_data: RenderedSliceData, redraw_canvas: bool = True
    ) -> None:
        """Callback to update objects when a new slice is received."""
        visual_object = self._visuals[slice_data.visual_id]
        visual_object.set_slice(slice_data=slice_data)

        if redraw_canvas:
            self.events.redraw_canvas.emit(
                CanvasRedrawRequest(scene_id=slice_data.scene_id)
            )
