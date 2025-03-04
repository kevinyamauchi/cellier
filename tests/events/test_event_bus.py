"""Test the event bus class."""

from psygnal import Signal

from cellier.events import EventBus
from cellier.models.visuals import PointsUniformMaterial, PointsVisual


class MockVisualGui:
    """Mock visual gui for testing.

    Attributes
    ----------
    n_update_calls : int
        Number of times the update method has been called.
    """

    update = Signal()

    def __init__(self, visual_id: str, visual_name: str):
        self.visual_id = visual_id
        self.visual_name = visual_name
        self.n_update_calls = 0

    def gui_update(self):
        """Emit the GUI updated signal."""
        self.update.emit({id: self.visual_id})

    def set_name(self, name: str):
        """Set the visual name."""
        self.visual_name = name
        self.update.emit(
            {
                "id": self.visual_id,
                "name": name,
                "controls_update_callback": self._on_model_update,
            }
        )

    def _on_model_update(self, event):
        """Callback for when the model updates."""
        self.n_update_calls += 1
        self.visual_name = event["name"]


def test_model_to_gui():
    """Test updating the GUI by making a change to the model."""

    # make a points visual model
    points_model = PointsVisual(
        name="points",
        data_store_id="dummy_id",
        material=PointsUniformMaterial(
            size=1.0,
            color=(1.0, 1.0, 1.0, 1.0),
        ),
    )

    # make a visual gui
    visual_gui = MockVisualGui(
        visual_id=points_model.id,
        visual_name=points_model.name,
    )

    event_bus = EventBus()

    # register the visual
    event_bus.visual.register_visual(points_model)

    # subscribe to the visual
    event_bus.visual.subscribe_to_visual(
        visual_id=points_model.id, callback=visual_gui._on_model_update
    )

    # register the visual gui
    event_bus.visual.register_controls(
        visual_id=points_model.id, signal=visual_gui.update
    )

    # subscribe the model to the gui
    event_bus.visual.subscribe_to_controls(
        visual_id=points_model.id, callback=points_model.update_state
    )

    # update the visual
    new_name = "new_name"
    # points_model.name = new_name
    points_model.update({"name": new_name})

    assert visual_gui.n_update_calls == 1
    assert visual_gui.visual_name == new_name


def test_gui_to_model():
    """Test updating the model by making a change to the GUI."""

    # make a points visual model
    points_model = PointsVisual(
        name="points",
        data_store_id="dummy_id",
        material=PointsUniformMaterial(
            size=1.0,
            color=(1.0, 1.0, 1.0, 1.0),
        ),
    )

    # make a visual gui
    visual_gui = MockVisualGui(
        visual_id=points_model.id,
        visual_name=points_model.name,
    )

    event_bus = EventBus()

    # register the visual
    event_bus.visual.register_visual(points_model)

    # subscribe to the visual
    event_bus.visual.subscribe_to_visual(
        visual_id=points_model.id, callback=visual_gui._on_model_update
    )

    # register the visual gui
    event_bus.visual.register_controls(
        visual_id=points_model.id, signal=visual_gui.update
    )

    # subscribe the model to the gui
    event_bus.visual.subscribe_to_controls(
        visual_id=points_model.id, callback=points_model.update_state
    )

    # update the name in the GUI
    new_name = "new_name"
    visual_gui.set_name(new_name)

    # make sure the model was updated
    assert points_model.name == new_name

    # make sure the GUI didn't refresh after the model update
    # (i.e., the signal was appropriately blocked)
    assert visual_gui.n_update_calls == 0
    assert visual_gui.visual_name == new_name

    # make sure the gui is unblocked
    points_model.name = "new_new_name"
    assert visual_gui.n_update_calls == 1
    assert visual_gui.visual_name == "new_new_name"
