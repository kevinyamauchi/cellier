import time

import numpy as np

from cellier.convenience import get_scene_with_dims_id
from cellier.models.data_manager import DataManager
from cellier.models.data_stores import PointsMemoryStore
from cellier.models.scene import (
    AxisAlignedRegionSelector,
    Canvas,
    CoordinateSystem,
    DimsManager,
    DimsState,
    OrbitCameraController,
    PerspectiveCamera,
    RangeTuple,
    Scene,
)
from cellier.models.viewer import SceneManager, ViewerModel
from cellier.models.visuals import PointsUniformMaterial, PointsVisual
from cellier.types import CoordinateSpace, DataResponse
from cellier.viewer_controller import CellierController


class SlicingValidator:
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

    def wait_for_slices(self, timeout: float = 5, error_on_timeout: bool = True):
        """Wait for a slice to be received."""
        self.n_slices_received = 0
        self.slices_received = []
        start_time = time.time()
        while self.n_slices_received == 0 and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        if self.n_slices_received == 0 and error_on_timeout:
            raise TimeoutError("Slice not received within the timeout period.")

        return

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
        """This is called when a new slice is received."""
        # store the slice
        self.slices_received.append(slice_response.data)

        # increment the counter
        self.n_slices_received += 1


def test_points_slicing(qtbot):
    # set up the coordinate system and dims manager
    coordinate_system = CoordinateSystem(name="default", axis_label=("x", "y", "z"))
    dims_manager = DimsManager(
        coordinate_system=coordinate_system,
        range=(RangeTuple(0, 11, 1), RangeTuple(0, 12, 1), RangeTuple(0, 13, 1)),
        selection=AxisAlignedRegionSelector(
            space_type=CoordinateSpace.WORLD,
            ordered_dims=(0, 1, 2),
            n_displayed_dims=2,
            index_selection=(0, slice(None, None, None), slice(None, None, None)),
        ),
    )

    # set up the points data
    coordinates = np.array([[0, 0, 0], [1, 2, 3], [10, 11, 12]])
    data_store = PointsMemoryStore(coordinates=coordinates)
    data = DataManager(stores={data_store.id: data_store})

    # set up the points visual
    points_material = PointsUniformMaterial(
        color=(1, 0, 0, 1),
        size=1,
    )
    points_visual = PointsVisual(
        name="points_visual", data_store_id=data_store.id, material=points_material
    )

    # make the canvas
    camera = PerspectiveCamera(
        width=110, height=110, controller=OrbitCameraController(enabled=True)
    )
    canvas = Canvas(camera=camera)

    # make the scene
    scene = Scene(
        dims=dims_manager, visuals=[points_visual], canvases={canvas.id: canvas}
    )
    scene_manager = SceneManager(scenes={scene.id: scene})

    viewer_model = ViewerModel(data=data, scenes=scene_manager)

    # make the controller
    viewer_controller = CellierController(model=viewer_model)

    # make the validator to check the slicing
    slicing_validator = SlicingValidator(
        dims_model=dims_manager, controller=viewer_controller
    )

    # wait for any slicing from the construction to finish
    slicing_validator.wait_for_slices(timeout=1, error_on_timeout=False)

    # update the selection
    dims_manager.selection.index_selection = (
        10,
        slice(None, None, None),
        slice(None, None, None),
    )

    slicing_validator.wait_for_slices(timeout=5)

    # check that the slicing was called once
    assert slicing_validator.n_dims_changed_events == 1

    # check that the correct slice was received
    assert slicing_validator.n_slices_received == 1
    np.testing.assert_allclose(
        slicing_validator.slices_received[0], np.array([[11, 12]])
    )
