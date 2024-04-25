"""Model for the viewer."""

import json
from typing import Dict

from psygnal import EventedModel
from pydantic_core import from_json

from cellier.models.data_stores.mesh import MeshMemoryStore
from cellier.models.data_streams.mesh import MeshSynchronousDataStream
from cellier.models.scene.scene import Scene


class DataManager(EventedModel):
    """Class to model all data_stores in the viewer.

    todo: move to separate module.
    todo: add discrimitive union
    """

    stores: Dict[str, MeshMemoryStore]
    streams: Dict[str, MeshSynchronousDataStream]


class SceneManager(EventedModel):
    """Class to model all scenes in the viewer."""

    scenes: Dict[str, Scene]


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
