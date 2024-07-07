"""Class for managing the data slicing."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING
from uuid import uuid4

from psygnal import Signal

from cellier.slicer.data_slice import DataSliceRequest, RenderedSliceData
from cellier.slicer.utils import world_slice_from_dims_manager

if TYPE_CHECKING:
    from cellier.models.viewer import ViewerModel


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

    def __init__(self, viewer_model: "ViewerModel"):
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
        """Callback for updating a scene when a dims model changes."""
        self.reslice_scene(scene_id=scene_id)

    def get_slice(self, slice_request: DataSliceRequest) -> RenderedSliceData:
        """Get a slice of data specified by the DataSliceRequest."""
        data_manager = self._viewer_model.data

        # get the data stream
        data_stream = data_manager.streams[slice_request.data_stream_id]

        # get the data store
        data_store = data_manager.stores[data_stream.data_store_id]

        # get the store slice
        data_store_slice = data_stream.get_data_store_slice(slice_request)

        return data_store.get_slice(data_store_slice)

    def reslice_scene(self, scene_id: str):
        """Update all objects in a given scene."""
        # get the DimsManager
        scene = self._viewer_model.scenes.scenes[scene_id]
        dims_manager = scene.dims

        # get the region to select in world coordinates
        # from the dims state
        world_slice = world_slice_from_dims_manager(dims_manager=dims_manager)

        for visual in scene.visuals:
            # get the transform
            # todo add transformation
            data_to_world_transform = None

            # apply the slice object to the data
            # todo move get_slice to DataManager?
            slice_response = self.get_slice(
                DataSliceRequest(
                    world_slice=world_slice,
                    resolution_level=0,
                    data_stream_id=visual.data_stream_id,
                    visual_id=visual.id,
                    scene_id=scene.id,
                    request_id=uuid4().hex,
                    data_to_world_transform=data_to_world_transform,
                )
            )

            # set the data
            self.events.new_slice.emit(slice_response)

    def reslice_all(self):
        """Reslice all scenes in the viewer."""
        for scene_id in self._viewer_model.scenes.scenes.keys():
            self.reslice_scene(scene_id=scene_id)


class AsynchronousDataSlicer:
    """Asynchronous data slicer class."""

    def __init__(self, viewer_model: "ViewerModel", max_workers: int = 3):
        self._viewer_model = viewer_model

        # add the events
        self.events = DataSlicerEvents()

        # attach the dims callbacks
        self._attach_dims_callbacks()

        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)

    def _attach_dims_callbacks(self) -> None:
        """Attach the callbacks to start the slice update when a dims model changes."""
        scene_manager = self._viewer_model.scenes

        for scene in scene_manager.scenes.values():
            dims_manager = scene.dims

            # connect to the events
            callback_function = partial(self._on_dims_update, scene_id=scene.id)
            dims_manager.events.all.connect(callback_function)

    def _on_dims_update(self, event, scene_id: str):
        """Callback for updating a scene when a dims model changes."""
        self.reslice_scene(scene_id=scene_id)

    def get_slice(self, slice_request: DataSliceRequest) -> RenderedSliceData:
        """Get a slice of data specified by the DataSliceRequest."""
        data_manager = self._viewer_model.data

        # get the data stream
        data_stream = data_manager.streams[slice_request.data_stream_id]

        # get the data store
        data_store = data_manager.stores[data_stream.data_store_id]

        # get the store slice
        data_store_slice = data_stream.get_data_store_slice(slice_request)

        return data_store.get_slice(data_store_slice)

    def reslice_scene(self, scene_id: str):
        """Update all objects in a given scene."""
        # get the DimsManager
        scene = self._viewer_model.scenes.scenes[scene_id]
        dims_manager = scene.dims

        # get the region to select in world coordinates
        # from the dims state
        world_slice = world_slice_from_dims_manager(dims_manager=dims_manager)

        for visual in scene.visuals:
            # get the transform
            # todo add transformation
            data_to_world_transform = None

            # apply the slice object to the data
            # todo move get_slice to DataManager?
            slice_response = self.get_slice(
                DataSliceRequest(
                    world_slice=world_slice,
                    resolution_level=0,
                    data_stream_id=visual.data_stream_id,
                    visual_id=visual.id,
                    scene_id=scene.id,
                    request_id=uuid4().hex,
                    data_to_world_transform=data_to_world_transform,
                )
            )

            # set the data
            self.events.new_slice.emit(slice_response)

    def reslice_all(self):
        """Reslice all scenes in the viewer."""
        for scene_id in self._viewer_model.scenes.scenes.keys():
            self.reslice_scene(scene_id=scene_id)
