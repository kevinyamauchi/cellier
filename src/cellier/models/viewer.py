"""Model for the viewer."""

import json
from typing import Dict

from psygnal import EventedModel
from pydantic_core import from_json

from cellier.models.data_manager import DataManager
from cellier.models.scene.scene import Scene


class SceneManager(EventedModel):
    """Class to model all scenes in the viewer.

    The keys are the scene ids.
    """

    scenes: Dict[str, Scene]

    def add_scene(self, scene: Scene) -> None:
        """Add a scene to the scene manager."""
        self.scenes[scene.id] = scene


class ViewerModel(EventedModel):
    """Class to model the viewer state."""

    data: DataManager
    scenes: SceneManager

    def to_json_file(self, file_path: str, indent: int = 2) -> None:
        """Save the viewer state as a JSON file."""
        with open(file_path, "w") as f:
            # serialize the model
            json.dump(self.model_dump(), f, indent=indent)

    @classmethod
    def from_json_file(cls, file_path: str):
        """Load a viewer from a JSON-formatted viewer state."""
        with open(file_path, "rb") as f:
            viewer_model = cls.model_validate(from_json(f.read(), allow_partial=False))
        return viewer_model
