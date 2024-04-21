"""Implementation of a viewer."""

from cellier.gui.constants import GuiFramework
from cellier.models.viewer import ViewerModel


class Viewer:
    """The main viewer class."""

    def __init__(
        self, model: ViewerModel, gui_framework: GuiFramework = GuiFramework.QT
    ):
        self._model = model
        self._gui_framework = gui_framework

        # Make the widget
        self._widget = self._construct_widget(viewer_model=self._model)

        # make the scene

    def _construct_widget(self, viewer_model: ViewerModel):
        """Make the viewer widget based on the requested gui framework.

        Parameters
        ----------
        viewer_model : ViewerModel
            THe viewer model to initialize the GUI from.
        """
        if self.gui_framework == GuiFramework.QT:
            # make a Qt widget
            from cellier.gui.qt.viewer import QtViewer

            return QtViewer(viewer_model=viewer_model)
        else:
            raise ValueError(f"Unsupported GUI framework: {self.gui_framework}")

    @property
    def gui_framework(self) -> GuiFramework:
        """The GUI framework used for this viewer."""
        return self._gui_framework


# class Viewer:
#     """Viewer class."""
#
#     def __init__(self, camera: PerspectiveCamera):
#         self._camera = camera
#         self.backend = ViewerBackend.from_models(camera=camera)
#
#         self._layer_list = EventedList()
#
#     @property
#     def camera(self) -> PerspectiveCamera:
#         """The camera model."""
#         return self._camera
#
#     @property
#     def layer_list(self) -> EventedList:
#         """The layers in the viewer."""
#         return self._layer_list
#
#     def add_mesh(
#         self,
#         data_source,
#         material,
#     ) -> None:
#         """Add a mesh to the viewer."""
#         self._add_visual(MeshVisualModel(data_source=data_source, material=material))
#
#     def _add_visual(self, visual_model):
#         # connect visual events
#         visual_model.connect_events()
#
#         # add the visual to the list
#         self.layer_list.append(visual_model)
#
#     def _on_layer_added(self, event):
#         self.backend.add_layer(event.value)
