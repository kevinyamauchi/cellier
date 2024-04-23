"""Model for the viewer."""

import json
from typing import List

from psygnal import EventedModel
from pydantic_core import from_json

from cellier.models.data_stores.base_data_store import BaseDataStore
from cellier.models.data_streams.base_data_stream import BaseDataStream
from cellier.models.scene.scene import Scene


class DataManager(EventedModel):
    """Class to model all data_stores in the viewer.

    todo: move to separate module.
    """

    stores: List[BaseDataStore]
    streams: List[BaseDataStream]


class SceneManager(EventedModel):
    """Class to model all scenes in the viewer."""

    scenes: List[Scene]


class ViewerModel(EventedModel):
    """Class to model the viewer state."""

    data: DataManager
    scenes: SceneManager

    def to_json_file(self, file_path: str) -> None:
        """Save the viewer state as a JSON file."""
        with open(file_path, "w") as f:
            # serialize the model
            json.dump(self.model_dump(), f)

    @classmethod
    def from_json_file(cls, file_path: str):
        """Load a viewer from a JSON-formatted viewer state."""
        with open(file_path, "rb") as f:
            viewer_model = cls.model_validate(from_json(f.read(), allow_partial=False))
        return viewer_model
