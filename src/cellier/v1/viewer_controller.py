"""Implementation of a viewer."""

from uuid import uuid4

import numpy as np

from cellier.gui.constants import GuiFramework
from cellier.models.data_stores.base_data_store import BaseDataStore
from cellier.models.nodes.base_node import BaseNode
from cellier.models.viewer import ViewerModel
from cellier.render._render_manager import (
    CameraState,
    CanvasRedrawRequest,
    RenderManager,
)
from cellier.slicer.slicer import (
    AsynchronousDataSlicer,
    DataSliceRequest,
    SlicerType,
    SynchronousDataSlicer,
)
from cellier.slicer.utils import world_slice_from_dims_manager
from cellier.util.chunk import generate_chunk_requests_from_frustum


class ViewerController:
    """The main viewer class."""

    def __init__(
        self,
        model: ViewerModel,
        gui_framework: GuiFramework = GuiFramework.QT,
        slicer_type: SlicerType = SlicerType.SYNCHRONOUS,
        widget_parent=None,
    ):
        self._model = model
        self._gui_framework = gui_framework

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
            self._slicer = SynchronousDataSlicer(viewer_model=self._model)
        elif slicer_type == SlicerType.ASYNCHRONOUS:
            self._slicer = AsynchronousDataSlicer()
        else:
            raise ValueError(f"Unknown slicer type: {slicer_type}")

        # connect events for rendering
        self._connect_render_events()

        # connect events for synchronizing the model and renderer
        self._connect_model_renderer_events()

        # update all of the slices
        self.reslice_all()

    def add_data_store(self, data_store: BaseDataStore):
        """Add a data store to the viewer."""
        self._model.data.add_data_store(data_store)

    def add_visual(self, visual_model: BaseNode, scene_id: str):
        """Add a visual to a scene."""
        # get the model
        scene = self._model.scenes.scenes[scene_id]
        scene.visuals.append(visual_model)

        # add the visual to the renderer
        self._render_manager.add_visual(visual_model=visual_model, scene_id=scene_id)

    def reslice_visual(self, scene_id: str, visual_id: str, canvas_id: str):
        """Reslice a specified visual."""
        # get the current dims
        scene = self._model.scenes.scenes[scene_id]
        visual = self._model.scenes.scenes[scene_id].get_visual_by_id(visual_id)

        # todo have the world slice passed in
        world_slice = world_slice_from_dims_manager(dims_manager=scene.dims)

        slice_request = DataSliceRequest(
            world_slice=world_slice,
            resolution_level=0,
            data_store_id=visual.data_store_id,
            visual_id=visual_id,
            scene_id=scene.id,
            request_id=uuid4().hex,
            data_to_world_transform=None,
        )

        # submit the request
        self._slicer.submit(
            request_list=[slice_request],
            data_store=self._model.data.stores[visual.data_store_id],
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

        print(f"index: {scale_index} scale: {data_store.scales[scale_index]}")

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
        print(f"shape: {texture_shape}, translation: {translation}")
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
        # callback to update the camera model on each draw
        self._render_manager.events.camera_updated.connect(self._update_camera_model)

    def _update_camera_model(self, camera_state: CameraState):
        scene_model = self._model.scenes.scenes[camera_state.scene_id]
        canvas_model = scene_model.canvases[camera_state.canvas_id]
        camera_model = canvas_model.camera
        camera_model.update(
            {
                "position": camera_state.position,
                "rotation": camera_state.rotation,
                "fov": camera_state.fov,
                "width": camera_state.width,
                "height": camera_state.height,
                "zoom": camera_state.zoom,
                "up_direction": camera_state.up_direction,
                "frustum": camera_state.frustum,
            }
        )

    def _on_canvas_redraw_event(self, event: CanvasRedrawRequest) -> None:
        """Called by the RenderManager when the canvas needs to be redrawn."""
        scene_model = self._model.scenes.scenes[event.scene_id]
        for canvas_model in scene_model.canvases.values():
            # refresh the canvas
            self._canvas_widgets[canvas_model.id].update()

    @property
    def gui_framework(self) -> GuiFramework:
        """The GUI framework used for this viewer."""
        return self._gui_framework
