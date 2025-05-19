"""Utilities for testing slicing."""

import time

from cellier.convenience import get_scene_with_dims_id
from cellier.models.scene import DimsManager, DimsState
from cellier.types import DataResponse
from cellier.viewer_controller import CellierController


class SlicingValidator:
    """A class to help validate the slicing of the data for visuals."""

    def __init__(self, dims_model: DimsManager, controller: CellierController):
        self._controller = controller

        # list to store the slices received
        self.slices_received = []

        # counter to check how many times dims update is called
        self.n_dims_changed_events = 0

        # counter to check how many slices received
        self.n_slices_received = 0

        # register the dims events
        self._controller.events.scene.register_dims(dims_model)

        # connect the redraw to the dims model
        self._controller.events.scene.subscribe_to_dims(
            dims_id=dims_model.id, callback=self._on_dims_update
        )

        # connect callback to the slicer's new slice event
        self._controller._slicer.events.new_slice.connect(self._on_new_slice)

    def wait_for_slices(
        self, n_slices: int = 1, timeout: float = 5, error_on_timeout: bool = True
    ):
        """Wait for a slice to be received.

        This blocks until an expected number of slices are received.

        Parameters
        ----------
        n_slices : int
            The number of slices to wait for.
            Default is 1 slice.
        timeout : float
            The maximum time to wait for the slices in seconds.
            Default is 5 seconds.
        error_on_timeout : bool
            If set to true, a TimeoutError is raised if the timeout is reached.
            Default is True.
        """
        self.n_slices_received = 0
        self.slices_received = []
        start_time = time.time()
        while (
            self.n_slices_received <= n_slices and (time.time() - start_time) < timeout
        ):
            time.sleep(0.1)

        if self.n_slices_received == 0 and error_on_timeout:
            raise TimeoutError("Slice not received within the timeout period.")

    def _on_dims_update(self, new_dims_state: DimsState):
        # perform the slicing when the dims are updated
        scene_model = get_scene_with_dims_id(
            viewer_model=self._controller._model,
            dims_id=new_dims_state.id,
        )
        self._controller.reslice_scene(scene_id=scene_model.id)

        # increment the counter
        self.n_dims_changed_events += 1

    def _on_new_slice(self, slice_response: DataResponse):
        """Callback that is called when a new slice is received."""
        # store the slice
        self.slices_received.append(slice_response.data)

        # increment the counter
        self.n_slices_received += 1
