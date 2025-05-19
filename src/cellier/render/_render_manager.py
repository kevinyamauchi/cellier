"""RenderManager class contains all the rendering and nodes code."""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict

import numpy as np
import pygfx
import pygfx as gfx
from psygnal import Signal
from pygfx.renderers import WgpuRenderer
from pylinalg import vec_transform, vec_unproject
from superqt import ensure_main_thread
from wgpu.gui import WgpuCanvasBase

from cellier.models.viewer import ViewerModel
from cellier.models.visuals.base import BaseVisual
from cellier.render._data_classes import (
    RendererCanvasMouseEvent,
    RendererVisualMouseEvent,
)
from cellier.render.cameras import construct_pygfx_camera_from_model
from cellier.render.utils import construct_pygfx_object
from cellier.types import (
    CameraControlsUpdateEvent,
    CameraId,
    CanvasId,
    DataResponse,
    MouseButton,
    MouseEventType,
    MouseModifiers,
    SceneId,
    VisualId,
)

logger = logging.getLogger(__name__)


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


# convert the pygfx mouse buttons to the Cellier mouse buttons
# https://jupyter-rfb.readthedocs.io/en/stable/events.html
pygfx_buttons_to_cellier_buttons = {
    0: MouseButton.NONE,
    1: MouseButton.LEFT,
    2: MouseButton.RIGHT,
    3: MouseButton.MIDDLE,
}

# convert the pygfx modifiers to the Cellier modifiers
# https://jupyter-rfb.readthedocs.io/en/stable/events.html
pygfx_modifiers_to_cellier_modifiers = {
    "Shift": MouseModifiers.SHIFT,
    "Control": MouseModifiers.CTRL,
    "Alt": MouseModifiers.ALT,
    "Meta": MouseModifiers.META,
}


class RenderManagerEvents:
    """Events for the RenderManager class."""

    redraw_canvas: Signal = Signal(CanvasRedrawRequest)
    camera_updated: Signal = Signal(CameraState)
    mouse_event: Signal = Signal(RendererVisualMouseEvent | RendererCanvasMouseEvent)


class RenderManager:
    """Class to manage the rendering."""

    def __init__(self, viewer_model: ViewerModel, canvases: Dict[str, WgpuCanvasBase]):
        # add the events
        self.events = RenderManagerEvents()

        # make each scene
        renderers = {}
        cameras = {}
        self._scenes = {}
        self._visuals = {}
        controllers = {}
        for scene_model in viewer_model.scenes.scenes.values():
            # make a scene
            scene = gfx.Scene()

            # add the background
            dark_gray = np.array((255, 255, 255, 255)) / 255
            light_gray = np.array((243, 243, 243, 255)) / 255
            background = gfx.Background.from_color(light_gray, dark_gray)
            scene.add(background)

            # todo add lighting config
            scene.add(gfx.AmbientLight())

            # todo add scene decorations config
            axes = gfx.AxesHelper(size=5, thickness=8)
            scene.add(axes)

            # store the scene
            scene_id = scene_model.id
            self._scenes.update({scene_id: scene})

            # get the camera and canvas IDs
            canvas_ids = []
            camera_ids = []
            for canvas in scene_model.canvases.values():
                canvas_ids.append(canvas.id)
                camera_ids.append(canvas.camera.id)

            # populate the scene
            for visual_model in scene_model.visuals:
                self.add_visual(
                    visual_model=visual_model,
                    scene_id=scene_model.id,
                    canvas_id=canvas_ids,
                    camera_id=camera_ids,
                )
                # add a bounding box
                # todo make configurable
                # box_world = gfx.BoxHelper(color="red")
                # box_world.set_transform_by_object(world_object)
                # scene.add(box_world)

            for canvas_id, canvas_model in scene_model.canvases.items():
                # make a renderer for each canvas
                canvas = canvases[canvas_id]
                renderer = WgpuRenderer(canvas)
                renderer.add_event_handler(
                    partial(
                        self._on_canvas_mouse_event,
                        canvas_id=canvas_id,
                        camera_id=canvas_model.camera.id,
                        event_type=MouseEventType.PRESS,
                    ),
                    "pointer_down",
                )

                # pointer up
                renderer.add_event_handler(
                    partial(
                        self._on_canvas_mouse_event,
                        canvas_id=canvas_id,
                        camera_id=canvas_model.camera.id,
                        event_type=MouseEventType.RELEASE,
                    ),
                    "pointer_up",
                )

                # pointer move
                renderer.add_event_handler(
                    partial(
                        self._on_canvas_mouse_event,
                        canvas_id=canvas_id,
                        camera_id=canvas_model.camera.id,
                        event_type=MouseEventType.MOVE,
                    ),
                    "pointer_move",
                )
                renderers.update({canvas_id: renderer})

                # make a camera and controller for each canvas
                camera, controller = construct_pygfx_camera_from_model(
                    camera_model=canvas_model.camera,
                )
                controller.register_events(renderer)

                # camera = gfx.PerspectiveCamera(width=110, height=110)
                # camera.show_object(scene)
                cameras.update({canvas_model.camera.id: camera})
                controllers.update({canvas_model.camera.id: controller})

                # connect a callback for the renderer
                # todo should this be outside the renderer?
                render_func = partial(
                    self.animate,
                    scene_id=scene_id,
                    canvas_id=canvas_id,
                    camera_id=canvas_model.camera.id,
                )
                canvas.request_draw(render_func)

        # store the values
        self._renderers = renderers
        self._cameras = cameras
        self._controllers = controllers

        self.render_calls = 0

    @property
    def renderers(self) -> Dict[CanvasId, WgpuRenderer]:
        """Dictionary of pygfx renderers.

        The key is the id property of the Canvas model the renderer
        belongs to.
        """
        return self._renderers

    @property
    def cameras(self) -> Dict[CameraId, pygfx.Camera]:
        """Dictionary of pygfx Cameras.

        The key is the id property of the Canvas model the Camera
        belongs to.
        """
        return self._cameras

    @property
    def camera_controllers(self) -> Dict[CameraId, pygfx.Controller]:
        """Dictionary of pygfx Camera controllers.

        The key is the id property of the Canvas model the controller
        belongs to.
        """
        return self._controllers

    @property
    def scenes(self) -> Dict[SceneId, gfx.Scene]:
        """Dictionary of pygfx Scenes.

        The key is the id of the Scene model the pygfx Scene belongs to.
        """
        return self._scenes

    @property
    def visuals(self) -> Dict[VisualId, gfx.WorldObject]:
        """The visuals in the RenderManager.

        The key is the id of the visual model.
        """
        return self._visuals

    def add_visual(
        self,
        visual_model: BaseVisual,
        scene_id: SceneId,
        canvas_id: list[CanvasId] | CanvasId,
        camera_id: list[CameraId] | CameraId,
    ):
        """Add a visual to a scene.

        Parameters
        ----------
        visual_model : BaseVisual
            The visual model to add.
        scene_id : SceneId
            The ID of the scene to add the visual to.
        canvas_id : list[CanvasId] | CanvasId
            The ID of the canvas to add the visual to.
        camera_id : list[CameraId] | CameraId
            The ID of the cameras of the scene. This must be the same
            shape as canvas_id.
        """
        # get the scene node
        scene = self._scenes[scene_id]

        # get the visual object
        world_object = construct_pygfx_object(visual_model=visual_model)

        # connect the mouse callback
        if isinstance(canvas_id, str):
            canvas_id = [canvas_id]
        if isinstance(camera_id, str):
            camera_id = [camera_id]
        for handler in world_object.callback_handlers:
            for canv_id, cam_id in zip(canvas_id, camera_id):
                # pointer down
                handler(
                    partial(
                        self._on_visual_mouse_event,
                        visual_id=visual_model.id,
                        canvas_id=canv_id,
                        camera_id=cam_id,
                        event_type=MouseEventType.PRESS,
                    ),
                    "pointer_down",
                )

                # pointer up
                handler(
                    partial(
                        self._on_visual_mouse_event,
                        visual_id=visual_model.id,
                        canvas_id=canv_id,
                        camera_id=cam_id,
                        event_type=MouseEventType.RELEASE,
                    ),
                    "pointer_up",
                )

                # pointer move
                handler(
                    partial(
                        self._on_visual_mouse_event,
                        visual_id=visual_model.id,
                        canvas_id=canv_id,
                        camera_id=cam_id,
                        event_type=MouseEventType.MOVE,
                    ),
                    "pointer_move",
                )

        # add the visual to the scene
        scene.add(world_object.node)

        self._visuals.update({visual_model.id: world_object})

    def add_visual_callback(
        self, visual_id: int, callback: Callable, callback_type: tuple[str, ...]
    ):
        """Add a callback to a visual."""
        visual = self.visuals[visual_id]
        visual.node.add_event_handler(callback, *callback_type)

    def remove_visual_callback(
        self, visual_id: int, callback: Callable, callback_type: tuple[str, ...]
    ):
        """Remove a callback from a visual."""
        visual = self.visuals[visual_id]
        visual.node.remove_event_handler(callback, *callback_type)

    def look_at_visual(
        self,
        visual_id: str,
        canvas_id: str,
        camera_id: str,
        scene_id: str,
        view_direction: tuple[float, float, float],
        up: tuple[float, float, float],
    ):
        """Set the camera to look at a specified visual.

        Parameters
        ----------
        visual_id : str
            The id of the visual to look at.
        canvas_id : str
            The id of the canvas to set.
        camera_id : str
            The id of the camera to set.
        scene_id : str
            The id of the scene the visual belongs to.
        view_direction : tuple[float, float, float]
            The direction to look at.
        up : tuple[float, float, float]
            The up direction.
        """
        # get the visual object
        visual = self._visuals[visual_id]

        # get the camera
        camera = self._cameras[camera_id]

        # set the camera to look at the visual
        camera.show_object(target=visual.node, view_dir=view_direction, up=up)

        self.animate(scene_id=scene_id, canvas_id=canvas_id, camera_id=camera_id)

    def animate(
        self, scene_id: SceneId, canvas_id: CanvasId, camera_id: CameraId
    ) -> None:
        """Callback to render a given canvas."""
        renderer = self.renderers[canvas_id]
        renderer.render(self.scenes[scene_id], self.cameras[camera_id])

        # Send event to update the cameras
        camera = self.cameras[camera_id]
        camera_state = camera.get_state()

        update_event = CameraControlsUpdateEvent(
            id=camera_id,
            state={
                "position": camera_state["position"],
                "rotation": camera_state["rotation"],
                "fov": camera_state["fov"],
                "up_direction": camera_state["reference_up"],
                "width": camera_state["width"],
                "height": camera_state["height"],
                "zoom": camera_state["zoom"],
                "frustum": camera.frustum,
                "controller": {"enabled": self._controllers[camera_id].enabled},
            },
            controls_update_callback=self._on_camera_model_update,
        )

        self.events.camera_updated.emit(update_event)
        #     CameraState(
        #         scene_id=scene_id,
        #         canvas_id=canvas_id,
        #         camera_id=camera.id,
        #         position=camera_state["position"],
        #         rotation=camera_state["rotation"],
        #         fov=camera_state["fov"],
        #         width=camera_state["width"],
        #         height=camera_state["height"],
        #         zoom=camera_state["zoom"],
        #         up_direction=camera_state["reference_up"],
        #         frustum=camera.frustum,
        #     )
        # )

        self.render_calls += 1
        logger.debug(f"render: {self.render_calls}")

    @ensure_main_thread
    def _on_new_slice(
        self, slice_data: DataResponse, redraw_canvas: bool = True
    ) -> None:
        """Callback to update objects when a new slice is received."""
        visual_object = self._visuals[slice_data.visual_id]
        visual_object.set_slice(slice_data=slice_data)

        if redraw_canvas:
            self.events.redraw_canvas.emit(
                CanvasRedrawRequest(scene_id=slice_data.scene_id)
            )

    def _on_canvas_mouse_event(
        self,
        event: gfx.PointerEvent,
        canvas_id: CanvasId,
        camera_id: CameraId,
        event_type: MouseEventType,
    ) -> None:
        """Process mouse callbacks from the canvas and rebroadcast."""
        # get the position of the click in screen coordinates
        position_screen = (event.x, event.y)
        renderer = self.renderers[canvas_id]
        renderer_size = renderer.logical_size

        # get the position of the click in NDC
        x = position_screen[0] / renderer_size[0] * 2 - 1
        y = -(position_screen[1] / renderer_size[1] * 2 - 1)
        pos_ndc = (x, y, 0)
        camera = self.cameras[camera_id]
        pos_ndc += vec_transform(camera.world.position, camera.camera_matrix)

        # get the position of the click in world space
        coordinate = vec_unproject(pos_ndc[:2], camera.camera_matrix)

        mouse_event = RendererCanvasMouseEvent(
            source_id=canvas_id,
            type=event_type,
            coordinate=coordinate,
            button=pygfx_buttons_to_cellier_buttons[event.button],
            modifiers=[
                pygfx_modifiers_to_cellier_modifiers[key] for key in event.modifiers
            ],
            pick_info=event.pick_info,
        )
        self.events.mouse_event(mouse_event)

    def _on_visual_mouse_event(
        self,
        event: gfx.PointerEvent,
        canvas_id: CanvasId,
        camera_id: CameraId,
        visual_id: VisualId,
        event_type: MouseEventType,
    ) -> None:
        """Process mouse callbacks from a visual and rebroadcast."""
        # get the position of the click in screen coordinates
        position_screen = (event.x, event.y)
        renderer = self.renderers[canvas_id]
        renderer_size = renderer.logical_size

        # get the position of the click in NDC
        x = position_screen[0] / renderer_size[0] * 2 - 1
        y = -(position_screen[1] / renderer_size[1] * 2 - 1)
        pos_ndc = (x, y, 0)
        camera = self.cameras[camera_id]
        pos_ndc += vec_transform(camera.world.position, camera.camera_matrix)

        # get the position of the click in world space
        coordinate = vec_unproject(pos_ndc[:2], camera.camera_matrix)

        mouse_event = RendererVisualMouseEvent(
            source_id=visual_id,
            type=event_type,
            coordinate=coordinate,
            button=pygfx_buttons_to_cellier_buttons[event.button],
            modifiers=[
                pygfx_modifiers_to_cellier_modifiers[key] for key in event.modifiers
            ],
            pick_info=event.pick_info,
        )
        self.events.mouse_event(mouse_event)

    def _on_camera_model_update(self, camera_state: CameraState):
        """Update the camera based on a change to the model."""
        depth_range = (
            camera_state.near_clipping_plane,
            camera_state.far_clipping_plane,
        )
        state_dict = {
            "position": camera_state.position,
            "rotation": camera_state.rotation,
            "reference_up": camera_state.up_direction,
            "fov": camera_state.fov,
            "width": camera_state.width,
            "height": camera_state.height,
            "zoom": camera_state.zoom,
            "depth_range": depth_range,
        }

        # get the camera
        camera = self.cameras[camera_state.id]

        # set the camera state
        # todo: consider checking if there any changes/update only changes
        camera.set_state(state_dict)

        # set the controller state
        controller = self._controllers[camera_state.id]
        controller.enabled = camera_state.controller.enabled
