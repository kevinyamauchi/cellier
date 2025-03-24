import numpy as np

from cellier.app.interactivity import LabelsPaintingManager, LabelsPaintingMode
from cellier.models.data_stores import ImageMemoryStore
from cellier.models.scene import PanZoomCameraController, PerspectiveCamera
from cellier.models.visuals import LabelsMaterial, MultiscaleLabelsVisual
from cellier.types import MouseButton, MouseCallbackData, MouseEventType


def test_labels_painting():
    data_store = ImageMemoryStore(data=np.zeros((30, 30, 30), dtype=np.uint16))
    labels_visual_model = MultiscaleLabelsVisual(
        name="labels_visual",
        data_store_id=data_store.id,
        material=LabelsMaterial(color_map="glasbey"),
        downscale_factors=[1],
    )
    camera = PerspectiveCamera(
        width=110, height=110, controller=PanZoomCameraController(enabled=True)
    )
    painting_manager = LabelsPaintingManager(
        labels_model=labels_visual_model, data_store=data_store, camera_model=camera
    )

    painting_manager.mode = LabelsPaintingMode.PAINT
    assert painting_manager.mode == LabelsPaintingMode.PAINT

    mouse_event = MouseCallbackData(
        visual_id=labels_visual_model.id,
        type=MouseEventType.PRESS,
        button=MouseButton.LEFT,
        modifiers=[],
        coordinate=np.array([10, 10, 10]),
        pick_info={},
    )

    # simulate a painting event
    painting_manager._on_mouse_press(mouse_event)
