import numpy as np

from cellier.models.data_manager import DataManager
from cellier.models.data_stores import ImageMemoryStore
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
from cellier.models.visuals import LabelsMaterial, MultiscaleLabelsVisual
from cellier.testing import SlicingValidator
from cellier.types import CoordinateSpace
from cellier.viewer_controller import CellierController


def test_labels_slicing(qtbot):
    """Test the slicing of image data."""
    # set up the coordinate system and dims manager
    coordinate_system = CoordinateSystem(
        name="default", axis_label=("t", "x", "y", "z")
    )
    dims_manager = DimsManager(
        coordinate_system=coordinate_system,
        range=(
            RangeTuple(0, 2, 1),
            RangeTuple(0, 21, 1),
            RangeTuple(0, 21, 1),
            RangeTuple(0, 21, 1),
        ),
        selection=AxisAlignedRegionSelector(
            space_type=CoordinateSpace.WORLD,
            ordered_dims=(0, 1, 2, 3),
            n_displayed_dims=3,
            index_selection=(
                0,
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
            ),
        ),
    )

    # set up the lines data
    image = np.zeros((3, 10, 10, 10), dtype=np.float32)
    image[1, :, :, :] = 1
    data_store = ImageMemoryStore(data=image)
    data = DataManager(stores={data_store.id: data_store})

    # set up the lines visual
    labels_material = LabelsMaterial(color_map="glasbey")
    labels_visual = MultiscaleLabelsVisual(
        name="labels_visual",
        data_store_id=data_store.id,
        material=labels_material,
        downscale_factors=[1],
    )

    # make the canvas
    camera = PerspectiveCamera(
        width=110, height=110, controller=OrbitCameraController(enabled=True)
    )
    canvas = Canvas(camera=camera)

    # make the scene
    scene = Scene(
        dims=dims_manager, visuals=[labels_visual], canvases={canvas.id: canvas}
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
        1,
        slice(None, None, None),
        slice(None, None, None),
        slice(None, None, None),
    )

    # wait for slicing to finish
    slicing_validator.wait_for_slices(timeout=5)

    # check that the slicing was called once
    assert slicing_validator.n_dims_changed_events == 1

    # check that the correct slice was received
    assert slicing_validator.n_slices_received == 1
    np.testing.assert_allclose(
        slicing_validator.slices_received[0], np.ones((10, 10, 10))
    )
