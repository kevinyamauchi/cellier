"""Class for managing the data slicing."""

from dataclasses import dataclass
from functools import partial
from uuid import uuid4

import numpy as np
from psygnal import Signal

from cellier.models.viewer import ViewerModel
from cellier.slicer.utils import world_slice_from_dims_manager


@dataclass
class RenderedSliceData:
    """Base class for rendered slice data classes.

    Note: all data should be in data coordinates.
    They will be transformed into world coordinates by
    the visual.

    Attributes
    ----------
    visual_id : str
        The UID of the visual to be updated.
    """

    visual_id: str


@dataclass
class RenderedMeshDataSlice(RenderedSliceData):
    """Data class for rendered mesh slice data.

    Attributes
    ----------
    visual_id : str
        The UID of the visual to be updated.
    vertices : np.ndarray
        The vertex coordinates of the new slice.
    faces  : np.ndarray
        The face indices of the new slices.

    """

    vertices: np.ndarray
    faces: np.ndarray


class DataSlicerEvents:
    """Event group for all data slicers.

    Attributes
    ----------
    new_slice : Signal
        This should be emitted when a slice is ready.
        The event should contain a RenderedSliceData object
        in the "data" field.
    """

    new_slice: Signal = Signal(RenderedSliceData)


class SynchronousDataSlicer:
    """A data slicer for synchronous slicing of data."""

    def __init__(self, viewer_model: ViewerModel):
        self._viewer_model = viewer_model

        # add the events
        self.events = DataSlicerEvents()

        # attach the dims callbacks
        self._attach_dims_callbacks()

    def _attach_dims_callbacks(self) -> None:
        """Attach the callbacks to start the slice update when a dims model changes."""
        scene_manager = self._viewer_model.scenes

        for scene in scene_manager.scenes.values():
            dims_manager = scene.dims

            # connect to the events
            callback_function = partial(self._on_dims_update, scene_id=scene.id)
            dims_manager.events.all.connect(callback_function)

    def _on_dims_update(self, event, scene_id: str):
        print(f"event: {event}")
        print(f"scene_id: {scene_id}")

        # get the DimsManager
        scene = self._viewer_model.scenes.scenes[scene_id]
        dims_manager = scene.dims

        # get the region to select in world coordinates
        # from the dims state
        world_slice = world_slice_from_dims_manager(dims_manager=dims_manager)

        for visual in scene.visuals:
            # get the transform
            data_to_world_transform = ""

            # apply the slice object to the data
            data_stream_id = visual.data_stream.id
            data_stream = self._viewer_model.data.streams[data_stream_id]
            slice_response = data_stream.get_slice(
                world_slice=world_slice,
                visual_id=visual.id,
                request_id=uuid4().hex,
                data_to_world_transform=data_to_world_transform,
            )

            # set the data
            self.events.new_slice.emit(slice_response)
