"""Implementation of a viewer."""

import logging
from typing import Callable

import numpy as np

from cellier.events import EventBus
from cellier.gui.constants import GuiFramework
from cellier.models.data_stores.base_data_store import BaseDataStore
from cellier.models.viewer import ViewerModel
from cellier.models.visuals.base import BaseVisual
from cellier.render._render_manager import (
    CanvasRedrawRequest,
    RenderManager,
)
from cellier.slicer.slicer import (
    AsynchronousDataSlicer,
    SlicerType,
)
from cellier.slicer.utils import (
    world_selected_region_from_dims,
)
from cellier.types import TilingMethod
from cellier.util.chunk import generate_chunk_requests_from_frustum

logger = logging.getLogger(__name__)


class CellierController:
    """Class to coordinate between the model and the renderer.

    This class does not handle the other aspects of the GUI.
    Applications should have a CellierController and synchronize their GUI with it
    via the events.
    """

    def __init__(
        self,
        model: ViewerModel,
        gui_framework: GuiFramework = GuiFramework.QT,
        slicer_type: SlicerType = SlicerType.ASYNCHRONOUS,
        widget_parent=None,
    ):
        self._model = model
        self._gui_framework = gui_framework

        # Make the event bus
        self._event_bus = EventBus(viewer_model=model)

        # Make the widget
        self._canvas_widgets = self._construct_canvas_widgets(
            viewer_model=self._model, parent=widget_parent
        )

        # make the scene
        self._render_manager = RenderManager(
            viewer_model=model, canvases=self._canvas_widgets
        )

        # make the slicer
        if slicer_type == SlicerType.SYNCHRONOUS:
            raise NotImplementedError("Synchronous slicer not implemented")
        elif slicer_type == SlicerType.ASYNCHRONOUS:
            self._slicer = AsynchronousDataSlicer()
        else:
            raise ValueError(f"Unknown slicer type: {slicer_type}")

        # connect events for rendering
        self._connect_render_events()

        # connect events for synchronizing the model and renderer
        self._connect_model_renderer_events()

        # connect events for mouse callbacks
        self._connect_mouse_events()

        # update all scenes
        self.reslice_all()

    @property
    def gui_framework(self) -> GuiFramework:
        """The GUI framework used for this viewer."""
        return self._gui_framework

    @property
    def events(self) -> EventBus:
        """The event bus for this viewer.

        All external components should only connect to the events
        in this EventBus object.
        """
        return self._event_bus

    def add_data_store(self, data_store: BaseDataStore):
        """Add a data store to the viewer."""
        self._model.data.add_data_store(data_store)

    def add_visual(self, visual_model: BaseVisual, scene_id: str):
        """Add a visual to a scene.

        In addition to adding the visual to the scene, this
        method also adds the visual model to the event bus.

        Parameters
        ----------
        visual_model : BaseVisual
            The visual model to add.
        scene_id : str
            The ID of the scene to add the visual to.
        """
        # add the model to the scene
        scene = self._model.scenes.scenes[scene_id]
        scene.visuals.append(visual_model)

        # get the canvas and camera ids
        canvas_ids = []
        camera_ids = []
        for canvas in scene.canvases.values():
            canvas_ids.append(canvas.id)
            camera_ids.append(canvas.camera.id)

        # add the visual to the renderer
        self._render_manager.add_visual(
            visual_model=visual_model,
            scene_id=scene_id,
            canvas_id=canvas_ids,
            camera_id=camera_ids,
        )

        # register the visual model with the eventbus
        self.events.visual.register_visual(visual=visual_model)

    def add_visual_callback(self, visual_id: str, callback: Callable):
        """Add a callback to a visual."""
        if visual_id not in self.events.mouse.visual_signals:
            # register the visual with the event bus
            self.events.mouse.register_visual(visual_id=visual_id)

        self.events.mouse.subscribe_to_visual(visual_id=visual_id, callback=callback)

    def remove_visual_callback(
        self,
        visual_id: str,
        callback: Callable,
    ):
        """Remove a callback from a visual."""
        self.events.mouse.visual_signals[visual_id].disconnect(
            callback, missing_ok=True
        )

    def look_at_visual(
        self,
        visual_id: str,
        view_direction: tuple[float, float, float],
        up: tuple[float, float, float],
    ):
        """Look at given visual.

        Parameters
        ----------
        visual_id : str
            The ID of the visual to look at.
        view_direction : tuple[float, float, float]
            The direction to set the camera view direction to.
        up : tuple[float, float, float]
            The direction to set the camera up direction to.
        """
        for scene in self._model.scenes.scenes.values():
            visual_model = scene.get_visual_by_id(visual_id)
            if visual_model is not None:
                for canvas_model in scene.canvases.values():
                    self._render_manager.look_at_visual(
                        visual_id=visual_id,
                        canvas_id=canvas_model.id,
                        camera_id=canvas_model.camera.id,
                        scene_id=scene.id,
                        view_direction=view_direction,
                        up=up,
                    )
                    self._canvas_widgets[canvas_model.id].update()
                return

    def reslice_visual(self, scene_id: str, visual_id: str, canvas_id: str):
        """Reslice a specified visual."""
        scene = self._model.scenes.scenes[scene_id]
        visual_model = scene.get_visual_by_id(visual_id)

        # get the region of the displayed dims being displayed
        selected_region = world_selected_region_from_dims(
            dims_manager=scene.dims,
            visual=visual_model,
        )

        # get the data requests
        data_store = self._model.data.stores[visual_model.data_store_id]
        requests = data_store.get_data_request(
            selected_region=selected_region,
            tiling_method=TilingMethod.NONE,
            scene_id=scene.id,
            visual_id=visual_model.id,
        )

        # submit the chunk requests
        self._slicer.submit(
            request_list=requests,
            data_store=data_store,
        )

    def reslice_visual_tiled(
        self, scene_id: str, visual_id: str, canvas_id: str
    ) -> None:
        """Reslice a specified using tiled rendering and frustum culling visual."""
        # get the current dims
        scene = self._model.scenes.scenes[scene_id]

        # get the region to select in world coordinates
        # from the dims state
        # todo deal with larger than 3D data.
        # world_slice = world_slice_from_dims_manager(dims_manager=dims_manager)

        # get the current camera and the frustum
        camera = scene.canvases[canvas_id].camera
        frustum_corners_world = camera.frustum

        # get the visual and data objects
        # todo add convenience to get visual by ID
        visual = scene.get_visual_by_id(visual_id)
        data_stream = self._model.data.streams[visual.data_stream_id]
        data_store = self._model.data.stores[data_stream.data_store_id]

        # get the frustum corners in the data local coordinates
        # todo: implement transforms
        frustum_corners_local = frustum_corners_world

        # get the current scale information
        renderer = self._render_manager.renderers[canvas_id]
        width_logical, height_logical = renderer.logical_size
        scale_index = data_store.determine_scale_from_frustum(
            frustum_corners=frustum_corners_local,
            width_logical=width_logical,
            height_logical=height_logical,
            method="logical_pixel_size",
        )

        # Convert the frustum corners to the scale coordinate system
        frustum_corners_scale = (
            frustum_corners_local / data_store.scales[scale_index]
        ) - data_store.translations[scale_index]

        logger.debug(f"index: {scale_index} scale: {data_store.scales[scale_index]}")

        # todo construct chunk corners using slicing
        # find the dims being displayed and then make the chunks
        chunk_corners = data_store.chunk_corners[scale_index]

        # construct the chunk request
        chunk_requests, texture_shape, translation_scale = (
            generate_chunk_requests_from_frustum(
                frustum_corners=frustum_corners_scale,
                chunk_corners=chunk_corners,
                scale_index=scale_index,
                scene_id=scene_id,
                visual_id=visual_id,
                mode="any",
            )
        )

        if len(chunk_requests) == 0:
            # no chunks to render
            return

        translation = (
            np.asarray(translation_scale) * data_store.scales[scale_index]
        ) + data_store.translations[scale_index]

        # pre allocate the data
        logger.debug(f"shape: {texture_shape}, translation: {translation}")
        visual = self._render_manager.visuals[visual_id]
        visual.preallocate_data(
            scale_index=scale_index,
            shape=texture_shape,
            chunk_shape=data_store.chunk_shapes[scale_index],
            translation=translation,
        )
        visual.set_scale_visible(scale_index)

        # submit the chunk requests to the slicer
        self._slicer.submit(request_list=chunk_requests, data_store=data_store)

    def reslice_scene(self, scene_id: str):
        """Update all objects in a given scene."""
        # get the Scene object
        scene = self._model.scenes.scenes[scene_id]

        # take the first canvas
        canvas_id = next(iter(scene.canvases))

        for visual in scene.visuals:
            self.reslice_visual(
                visual_id=visual.id, scene_id=scene.id, canvas_id=canvas_id
            )

    def reslice_all(
        self,
    ) -> None:
        """Reslice all visuals."""
        for scene_id in self._model.scenes.scenes.keys():
            self.reslice_scene(scene_id=scene_id)

    def _construct_canvas_widgets(self, viewer_model: ViewerModel, parent=None):
        """Make the canvas widgets based on the requested gui framework.

        Parameters
        ----------
        viewer_model : ViewerModel
            The viewer model to initialize the GUI from.
        parent : Optional
            The parent widget to assign to the constructed canvas widgets.
            The default value is None.
        """
        if self.gui_framework == GuiFramework.QT:
            # make a Qt widget
            from cellier.gui.qt.utils import construct_qt_canvases_from_model

            return construct_qt_canvases_from_model(
                viewer_model=viewer_model, parent=parent
            )
        else:
            raise ValueError(f"Unsupported GUI framework: {self.gui_framework}")

    def _connect_render_events(self):
        """Connect callbacks to the render events."""
        # add a callback to update the scene when a new slice is available
        self._slicer.events.new_slice.connect(self._render_manager._on_new_slice)

        # add a callback to refresh the canvas when the scene has been updated
        self._render_manager.events.redraw_canvas.connect(self._on_canvas_redraw_event)

    def _connect_model_renderer_events(self):
        """Connect callbacks to keep the model and the renderer in sync."""
        # register the camera and controls with the event bus
        for scene in self._model.scenes.scenes.values():
            for canvas in scene.canvases.values():
                # register the camera model with the event bus
                camera_model = canvas.camera
                self.events.scene.register_camera(
                    camera_model=camera_model,
                )

                camera_model_id = canvas.camera.id
                # subscribe the renderer to the camera model
                self.events.scene.subscribe_to_camera(
                    camera_id=camera_model_id,
                    callback=self._render_manager._on_camera_model_update,
                )

                # register the renderer camera to the event bus
                self.events.scene.register_camera_controls(
                    camera_id=camera_model_id,
                    signal=self._render_manager.events.camera_updated,
                )

                # register the camera model to the renderer event
                self.events.scene.subscribe_to_camera_controls(
                    camera_id=camera_model_id, callback=camera_model.update_state
                )

    def _connect_mouse_events(self):
        """Register all visuals and renderers with the mouse events bus."""
        # register all visuals
        # for visual_id, visual in self._render_manager.visuals.items():
        #     if visual_id not in self.events.mouse.visual_signals:
        #         self.events.mouse.register_visual(
        #             visual_id=visual_id,
        #             callback_handlers=visual.callback_handlers
        #         )
        self._render_manager.events.mouse_event.connect(
            self.events.mouse._on_mouse_event
        )

    def _on_canvas_redraw_event(self, event: CanvasRedrawRequest) -> None:
        """Called by the RenderManager when the canvas needs to be redrawn."""
        scene_model = self._model.scenes.scenes[event.scene_id]
        for canvas_model in scene_model.canvases.values():
            # refresh the canvas
            self._canvas_widgets[canvas_model.id].update()
