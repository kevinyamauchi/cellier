from psygnal import Signal

from cellier.events import EventBus
from cellier.models.scene import (
    AxisAlignedRegionSelector,
    CoordinateSystem,
    DimsManager,
    DimsState,
    RangeTuple,
)
from cellier.types import CoordinateSpace, DimsControlsUpdateEvent


class MockDimsGui:
    """Mock dims controls for testing."""

    currentIndexChanged = Signal(DimsControlsUpdateEvent)

    def __init__(self, dims_id: str):
        self.dims_id = dims_id
        self.n_update_calls = 0
        self.n_displayed_dims = 3

    def update_n_dims_displayed(self, n_dims_displayed: int):
        """Update the point in the dims GUI."""
        self.n_dims_displayed = n_dims_displayed
        update_event = DimsControlsUpdateEvent(
            id=self.dims_id,
            state={"selection": {"n_displayed_dims": n_dims_displayed}},
            controls_update_callback=self._on_dims_state_changed,
        )
        self.currentIndexChanged(update_event)

    def _on_dims_state_changed(self, new_state: DimsState):
        self.n_displayed_dims = new_state.selection.n_displayed_dims
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
    selection = AxisAlignedRegionSelector(
        space_type=CoordinateSpace.WORLD,
        ordered_dims=(0, 1, 2),
        n_displayed_dims=3,
        index_selection=(0, slice(0, 10, 1), slice(None, None, None)),
    )
    dims_3d = DimsManager(
        range=data_range,
        coordinate_system=coordinate_system_3d,
        selection=selection,
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

    new_n_dims_displayed = 2
    dims_3d.selection.n_displayed_dims = new_n_dims_displayed

    assert dims_controls.n_displayed_dims == new_n_dims_displayed
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

    # verify the n displayed dims is 3
    assert dims_3d.selection.n_displayed_dims == 3

    new_n_dims_displayed = 2
    dims_controls.update_n_dims_displayed(new_n_dims_displayed)

    assert dims_3d.selection.n_displayed_dims == new_n_dims_displayed

    # verify the update didn't trigger a callback on the GUI
    # (i.e., no signal "bounce back")
    assert dims_controls.n_update_calls == 0
