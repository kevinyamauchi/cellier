"""Test fixtures for Cellier."""

import numpy as np
import pytest

from cellier.models.data_manager import DataManager
from cellier.models.data_stores import PointsMemoryStore
from cellier.models.scene import (
    AxisAlignedRegionSelector,
    Canvas,
    CoordinateSystem,
    DimsManager,
    PanZoomCameraController,
    PerspectiveCamera,
    RangeTuple,
    Scene,
)
from cellier.models.viewer import SceneManager, ViewerModel
from cellier.models.visuals import PointsUniformMaterial, PointsVisual
from cellier.types import CoordinateSpace


@pytest.fixture
def viewer_model_2d_points() -> dict[str, ViewerModel | PointsVisual | Canvas]:
    """Make a simple 2D scene with a points visual."""

    # make the points data store
    point_coordinates = np.array([[0, 0], [1, 1], [2, 2]])
    points_store = PointsMemoryStore(
        coordinates=point_coordinates,
    )

    # make the data manager
    data = DataManager(stores={points_store.id: points_store})

    # make the points visual
    material = PointsUniformMaterial(size=1, color=(1, 0, 0, 1))
    points_visual = PointsVisual(
        name="points_2d",
        data_store_id=points_store.id,
        material=material,
    )

    # make the 2D scene coordinate system
    coordinate_system = CoordinateSystem(name="default", axis_label=("x", "y"))
    dims_2d = DimsManager(
        coordinate_system=coordinate_system,
        range=(
            RangeTuple(0, 10, 1),
            RangeTuple(0, 10, 1),
        ),
        selection=AxisAlignedRegionSelector(
            space_type=CoordinateSpace.WORLD,
            ordered_dims=(0, 1),
            n_displayed_dims=2,
            index_selection=(0, slice(0, 10, 1)),
        ),
    )

    # make the 2D canvas
    camera_2d = PerspectiveCamera(
        fov=0, controller=PanZoomCameraController(enabled=True)
    )
    canvas_2d = Canvas(camera=camera_2d)

    # make the scene
    scene_2d = Scene(
        dims=dims_2d, visuals=[points_visual], canvases={canvas_2d.id: canvas_2d}
    )

    # make the scene manager
    scene_manager = SceneManager(
        scenes={
            scene_2d.id: scene_2d,
        }
    )

    # make the viewer model
    return {
        "viewer": ViewerModel(data=data, scenes=scene_manager),
        "visual": points_visual,
        "canvas": canvas_2d,
    }
