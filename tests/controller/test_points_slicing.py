import numpy as np

from cellier.models.data_manager import DataManager
from cellier.models.data_stores import PointsMemoryStore
from cellier.models.scene import (
    AxisAlignedRegionSelector,
    Canvas,
    CoordinateSystem,
    DimsManager,
    OrbitCameraController,
    PerspectiveCamera,
    RangeTuple,
    Scene,
)
from cellier.models.viewer import SceneManager, ViewerModel
from cellier.models.visuals import PointsUniformMaterial, PointsVisual
from cellier.testing import SlicingValidator
from cellier.types import CoordinateSpace
from cellier.viewer_controller import CellierController


def test_points_slicing(qtbot):
    """Test the slicing of points data."""
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
