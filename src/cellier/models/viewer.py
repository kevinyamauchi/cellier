"""Model for the viewer."""

from typing import List

from psygnal import EventedModel

from cellier.models.data_stores.base_data_store import BaseDataStore
from cellier.models.data_streams.base_data_stream import BaseDataStream
from cellier.models.scene.scene import Scene


class DataManager(EventedModel):
    """Class to model all data in the viewer.

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
