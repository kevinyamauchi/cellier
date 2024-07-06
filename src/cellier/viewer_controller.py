"""Implementation of a viewer."""

from cellier.gui.constants import GuiFramework
from cellier.models.viewer import ViewerModel
from cellier.render.render_manager import CanvasRedrawRequest, RenderManager
from cellier.slicer.slicer import SynchronousDataSlicer


class ViewerController:
    """The main viewer class."""

    def __init__(
        self,
        model: ViewerModel,
        gui_framework: GuiFramework = GuiFramework.QT,
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
        self._slicer = SynchronousDataSlicer(viewer_model=self._model)

        # connect events
        self._connect_render_events()

        # connect events for synchronizing the model and renderer
        # self._connect_model_renderer_events()

        # update all of the slices
        self._slicer.reslice_all()

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

    # def _connect_model_renderer_events(self):
    #     """Connect callbacks to keep the model and the renderer in sync."""
    #     # callback to update the camera model on each draw
    #     for canvas_id, renderer in self._render_manager._renderers.items():
    #         #

    def _on_canvas_redraw_event(self, event: CanvasRedrawRequest) -> None:
        """Called by the RenderManager when the canvas needs to be redrawn."""
        scene_model = self._model.scenes.scenes[event.scene_id]
        for canvas_model in scene_model.canvases:
            # refresh the canvas
            self._canvas_widgets[canvas_model.id].update()

    @property
    def gui_framework(self) -> GuiFramework:
        """The GUI framework used for this viewer."""
        return self._gui_framework
