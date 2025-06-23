import numpy as np

from cellier.models.data_manager import DataManager
from cellier.models.data_stores import LinesMemoryStore
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
from cellier.models.visuals import (
    LinesUniformAppearance,
    LinesVertexColorAppearance,
    LinesVisual,
)
from cellier.testing import SlicingValidator
from cellier.types import CoordinateSpace
from cellier.viewer_controller import CellierController


def test_lines_slicing(qtbot):
    """Test the slicing of lines data."""
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
    coordinates = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 10, 10],
            [1, 1, 1, 1],
            [1, 1, 11, 11],
            [0, 10, 10, 10],
            [0, 10, 20, 20],
        ]
    )
    data_store = LinesMemoryStore(coordinates=coordinates)
    data = DataManager(stores={data_store.id: data_store})

    # set up the lines visual
    lines_materials = LinesUniformAppearance(
        size=1,
        color=(1, 0, 0, 1),
        opacity=1.0,
    )
    lines_visual = LinesVisual(
        name="lines_visual", data_store_id=data_store.id, appearance=lines_materials
    )

    # make the canvas
    camera = PerspectiveCamera(
        width=110, height=110, controller=OrbitCameraController(enabled=True)
    )
    canvas = Canvas(camera=camera)

    # make the scene
    scene = Scene(
        dims=dims_manager, visuals=[lines_visual], canvases={canvas.id: canvas}
    )
    scene_manager = SceneManager(scenes={scene.id: scene})

    viewer_model = ViewerModel(data=data, scenes=scene_manager)

    # make the controller
    viewer_controller = CellierController.from_viewer_model(
        viewer_model=viewer_model, slice_all=True
    )

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
        slicing_validator.slices_received[0].data,
        np.array(
            [
                [1, 1, 1],
                [1, 11, 11],
            ]
        ),
    )


def test_lines_color_slicing(qtbot):
    """Test the slicing of lines data with colors."""
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

    # set up the lines data with colors
    coordinates = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 10, 10],
            [1, 1, 1, 1],
            [1, 1, 11, 11],
            [0, 10, 10, 10],
            [0, 10, 20, 20],
        ]
    )
    colors = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 0.5, 0.5, 1.0],
            [0.5, 1.5, 1.5, 1.0],
        ]
    )
    data_store = LinesMemoryStore(coordinates=coordinates, colors=colors)
    data = DataManager(stores={data_store.id: data_store})

    # set up the lines visual
    lines_materials = LinesVertexColorAppearance(
        size=1,
        opacity=1.0,
    )
    lines_visual = LinesVisual(
        name="lines_visual", data_store_id=data_store.id, appearance=lines_materials
    )

    # make the canvas
    camera = PerspectiveCamera(
        width=110, height=110, controller=OrbitCameraController(enabled=True)
    )
    canvas = Canvas(camera=camera)

    # make the scene
    scene = Scene(
        dims=dims_manager, visuals=[lines_visual], canvases={canvas.id: canvas}
    )
    scene_manager = SceneManager(scenes={scene.id: scene})

    viewer_model = ViewerModel(data=data, scenes=scene_manager)

    # make the controller
    viewer_controller = CellierController.from_viewer_model(
        viewer_model=viewer_model, slice_all=True
    )

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
        slicing_validator.slices_received[0].data,
        np.array(
            [
                [1, 1, 1],
                [1, 11, 11],
            ]
        ),
    )
    np.testing.assert_allclose(
        slicing_validator.slices_received[0].colors,
        np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
            ]
        ),
    )
