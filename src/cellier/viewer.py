"""Implementation of a viewer."""

from psygnal.containers import EventedList

from cellier.models.scene.cameras import PerspectiveCamera
from cellier.models.visuals.mesh_visual import MeshVisualModel
from cellier.viewer_backend import ViewerBackend


class Viewer:
    """Viewer class."""

    def __init__(self, camera: PerspectiveCamera):
        self._camera = camera
        self.backend = ViewerBackend.from_models(camera=camera)

        self._layer_list = EventedList()

    @property
    def camera(self) -> PerspectiveCamera:
        """The camera model."""
        return self._camera

    @property
    def layer_list(self) -> EventedList:
        """The layers in the viewer."""
        return self._layer_list

    def add_mesh(
        self,
        data_source,
        material,
    ) -> None:
        """Add a mesh to the viewer."""
        self._add_visual(MeshVisualModel(data_source=data_source, material=material))

    def _add_visual(self, visual_model):
        # connect visual events
        visual_model.connect_events()

        # add the visual to the list
        self.layer_list.append(visual_model)

    def _on_layer_added(self, event):
        self.backend.add_layer(event.value)
