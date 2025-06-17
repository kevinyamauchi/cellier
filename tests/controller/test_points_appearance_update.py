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
from cellier.models.visuals import PointsUniformAppearance, PointsVisual
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
    points_material = PointsUniformAppearance(
        color=(1, 0, 0, 1),
        size=1,
    )
    points_visual = PointsVisual(
        name="points_visual", data_store_id=data_store.id, appearance=points_material
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

    # verify the renderer visual implementation is visible
    points_renderer_implementation = viewer_controller._render_manager._visuals[
        points_visual.id
    ]
    assert points_renderer_implementation.node.visible

    # set visible to false and verify
    points_visual.appearance.visible = False
    assert not points_renderer_implementation.node.visible
