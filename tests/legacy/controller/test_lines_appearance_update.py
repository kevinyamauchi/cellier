import numpy as np

from cellier._legacy.models.data_manager import DataManager
from cellier._legacy.models.data_stores import LinesMemoryStore
from cellier._legacy.models.scene import (
    AxisAlignedRegionSelector,
    Canvas,
    CoordinateSystem,
    DimsManager,
    OrbitCameraController,
    PerspectiveCamera,
    RangeTuple,
    Scene,
)
from cellier._legacy.models.viewer import SceneManager, ViewerModel
from cellier._legacy.models.visuals import LinesUniformAppearance, LinesVisual
from cellier._legacy.types import CoordinateSpace
from cellier._legacy.viewer_controller import CellierController


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
        size=1, color=(1, 0, 0, 1), opacity=1.0, visible=True
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
    viewer_controller = CellierController(model=viewer_model)

    # make sure the renderer visual is visible
    lines_renderer_implementation = viewer_controller._render_manager._visuals[
        lines_visual.id
    ]
    assert lines_renderer_implementation.node.visible

    # set visibility to False and verify
    lines_visual.appearance.visible = False
    assert not lines_renderer_implementation.node.visible
