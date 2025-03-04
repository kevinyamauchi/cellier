"""Class for managing the data slicing."""

from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Dict, List
from uuid import uuid4

from psygnal import Signal

from cellier.slicer.data_slice import DataSliceRequest, RenderedSliceData
from cellier.slicer.utils import world_slice_from_dims_manager

if TYPE_CHECKING:
    from cellier.models.viewer import ViewerModel


class SlicerType(Enum):
    """Enum for supported slicer types.

    SYNCHRONOUS will use SynchronousDataSlicer
    ASYNCHRONOUS will use AsynchronousDataSlicer
    """

    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"


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

        # get the data store
        data_store = data_manager.stores[slice_request.data_store_id]

        return data_store.get_slice(slice_request)

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
                    data_store_id=visual.data_store_id,
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

    def __init__(self, max_workers: int = 2):
        # add the events
        self.events = DataSlicerEvents()

        # Storage for pending futures.
        # The key is the visual id the slice request originated from.
        self._pending_futures: Dict[str, list[Future[RenderedSliceData]]] = {}
        self._futures_to_ignore: List[Future[RenderedSliceData]] = []

        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, request_list, data_store):
        """Submit a request for a slice."""
        if len(request_list) == 0:
            print("no chunks")
            return

        visual_id = request_list[0].visual_id
        if visual_id in self._pending_futures:
            # try to cancel the future
            futures_to_cancel_list = self._pending_futures.pop(visual_id)

            for future_to_cancel in futures_to_cancel_list:
                # cancel each future in the list
                cancelled = future_to_cancel.cancel()

                if not cancelled:
                    # sometimes futures can't be canceled
                    # we store a reference to this future so we can ignore it
                    # when the result comes in.
                    self._futures_to_ignore.append(future_to_cancel)

        slice_futures_list = []
        for request in request_list:
            slice_future = self._thread_pool.submit(data_store.get_slice, request)

            # add the callback to send the data when the slice is received
            slice_future.add_done_callback(self._on_slice_response)
            slice_futures_list.append(slice_future)

        # store the future
        self._pending_futures[visual_id] = slice_futures_list

    def _on_slice_response(self, future: Future[RenderedSliceData]):
        if future.cancelled():
            # if the future was cancelled, return early
            return

        if future in self._futures_to_ignore:
            print("ignoring")
            self._futures_to_ignore.remove(future)
            return

        # get the data
        slice_response = future.result()

        # remove the future from the pending dict
        visual_id = slice_response.visual_id
        if visual_id in self._pending_futures:
            del self._pending_futures[visual_id]

        # emit the slice
        self.events.new_slice.emit(slice_response)
