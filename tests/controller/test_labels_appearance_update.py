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
from cellier.models.visuals import LabelsAppearance, MultiscaleLabelsVisual
from cellier.types import CoordinateSpace
from cellier.viewer_controller import CellierController


def test_labels_appearance_update(qtbot):
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

    # set up the labels data
    image = np.zeros((3, 10, 10, 10), dtype=np.float32)
    image[1, :, :, :] = 1
    data_store = ImageMemoryStore(data=image)
    data = DataManager(stores={data_store.id: data_store})

    # set up the lines visual
    labels_material = LabelsAppearance(color_map="glasbey", visible=True)
    labels_visual = MultiscaleLabelsVisual(
        name="labels_visual",
        data_store_id=data_store.id,
        appearance=labels_material,
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

    # get the renderer visual
    labels_renderer_implementation = viewer_controller._render_manager._visuals[
        labels_visual.id
    ]

    # check that the labels visual is visible
    assert labels_renderer_implementation.node.visible

    # set the labels to not visible and verify
    labels_visual.appearance.visible = False
    assert not labels_renderer_implementation.node.visible
