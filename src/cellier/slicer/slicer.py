"""Class for managing the data slicing."""

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from typing import TYPE_CHECKING, Dict, List

from psygnal import Signal

from cellier.types import DataResponse

if TYPE_CHECKING:
    from cellier.types import VisualId

logger = logging.getLogger(__name__)


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

    new_slice: Signal = Signal(DataResponse)


class AsynchronousDataSlicer:
    """Asynchronous data slicer class."""

    def __init__(self, max_workers: int = 2):
        # add the events
        self.events = DataSlicerEvents()

        # Storage for pending futures.
        # The key is the visual id the slice request originated from.
        self._pending_futures: Dict["VisualId", list[Future[DataResponse]]] = {}
        self._futures_to_ignore: List[Future[DataResponse]] = []

        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, request_list, data_store):
        """Submit a request for a slice."""
        if len(request_list) == 0:
            logger.debug("no chunks submitted for slicing.")
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
            slice_future = self._thread_pool.submit(data_store.get_data, request)

            # add the callback to send the data when the slice is received
            slice_future.add_done_callback(self._on_slice_response)
            slice_futures_list.append(slice_future)

        # store the future
        self._pending_futures[visual_id] = slice_futures_list

    def _on_slice_response(self, future: Future[DataResponse]):
        if future.cancelled():
            # if the future was cancelled, return early
            return

        if future in self._futures_to_ignore:
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
