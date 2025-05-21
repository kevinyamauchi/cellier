"""Model for the viewer."""

from typing import Dict

from psygnal import EmissionInfo, EventedModel, Signal
from pydantic_core import from_json

from cellier.models.data_manager import DataManager
from cellier.models.scene import DimsManager, DimsState, Scene


class SceneManager(EventedModel):
    """Class to model all scenes in the viewer.

    The keys are the scene ids.
    """

    scenes: Dict[str, Scene]

    def model_post_init(self, __context):
        """Called after the model is initialized.

        This is inherited from the Pydantic BaseModel.
        Currently, this is used to connect the scene events to the scene manager.
        """
        for scene in self.scenes.values():
            # connect the scene events to the scene manager
            self._connect_scene_events(scene=scene)

    def add_scene(self, scene: Scene) -> None:
        """Add a scene to the scene manager."""
        self.scenes[scene.id] = scene

        # connect the scene events to the scene manager
        self._connect_scene_events(scene=scene)

    def _connect_scene_events(self, scene: Scene) -> None:
        """Connect the scene events to the scene manager."""
        # connect the dims events to the scene manager's dims event
        scene.dims.events.all.connect(self._on_dims_updated)

    def _on_dims_updated(self, event: EmissionInfo):
        """Handle the dims update event."""
        dims_manager: DimsManager = Signal.sender()
        dims_state: DimsState = dims_manager.to_state()
        self.events.scenes.emit(dims_state)


class ViewerModel(EventedModel):
    """Class to model the viewer state."""

    data: DataManager
    scenes: SceneManager

    def to_json_file(self, file_path: str, indent: int = 2) -> None:
        """Save the viewer state as a JSON file."""
        with open(file_path, "w") as f:
            # serialize the model
            f.write(self.model_dump_json(indent=indent))

    @classmethod
    def from_json_file(cls, file_path: str):
        """Load a viewer from a JSON-formatted viewer state."""
        with open(file_path, "rb") as f:
            viewer_model = cls.model_validate(from_json(f.read(), allow_partial=False))
        return viewer_model
