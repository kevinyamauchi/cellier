from psygnal import Signal

from cellier.events import DimsControlsUpdateEvent, EventBus
from cellier.models.scene import CoordinateSystem, DimsManager, DimsState, RangeTuple


class MockDimsGui:
    """Mock dims controls for testing."""

    currentIndexChanged = Signal(DimsControlsUpdateEvent)

    def __init__(self, dims_id: str):
        self.dims_id = dims_id
        self.n_update_calls = 0
        self.point = (0, 0, 0)

    def update_point(self, point: tuple[int, int, int]):
        """Update the point in the dims GUI."""
        self.point = point
        update_event = DimsControlsUpdateEvent(
            id=self.dims_id,
            state={"point": point},
            controls_update_callback=self._on_dims_state_changed,
        )
        self.currentIndexChanged(update_event)

    def _on_dims_state_changed(self, new_state: DimsState):
        self.point = new_state.point
        self.n_update_calls += 1


def setup_dims_model_and_controls() -> tuple[DimsManager, MockDimsGui]:
    """Create a dims model and controls for testing."""
    # make the dims model
    data_range = (
        RangeTuple(start=0, stop=250, step=1),
        RangeTuple(start=0, stop=250, step=1),
        RangeTuple(start=0, stop=250, step=1),
    )
    coordinate_system_3d = CoordinateSystem(
        name="scene_3d", axis_labels=("z", "y", "x")
    )
    dims_3d = DimsManager(
        point=(0, 0, 0),
        margin_negative=(0, 0, 0),
        margin_positive=(0, 0, 0),
        range=data_range,
        coordinate_system=coordinate_system_3d,
        displayed_dimensions=(1, 2),
    )

    # make the dims controls
    dims_controls = MockDimsGui(dims_id=dims_3d.id)

    return dims_3d, dims_controls


def test_dims_model_to_gui():
    """Test updating the GUI by making a change to the dims model."""

    # set up the dims model and controls
    dims_3d, dims_controls = setup_dims_model_and_controls()

    # make the event bus
    event_bus = EventBus()

    # add the dims model and controls to the event buss
    event_bus.scene.add_dims_with_controls(
        dims_model=dims_3d,
        dims_controls=dims_controls,
    )

    new_dims_point = (10, 0, 0)
    dims_3d.point = new_dims_point

    assert dims_controls.point == new_dims_point
    assert dims_controls.n_update_calls == 1


def test_gui_to_dims_model():
    """Test updating the dims model by making a change to the GUI."""
    # set up the dims model and controls
    dims_3d, dims_controls = setup_dims_model_and_controls()

    # make the event bus
    event_bus = EventBus()

    # add the dims model and controls to the event buss
    event_bus.scene.add_dims_with_controls(
        dims_model=dims_3d,
        dims_controls=dims_controls,
    )

    new_dims_point = (10, 0, 0)
    dims_controls.update_point(new_dims_point)

    assert dims_3d.point == new_dims_point
    assert dims_controls.n_update_calls == 0
