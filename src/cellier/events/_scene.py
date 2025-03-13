import logging
from dataclasses import dataclass
from typing import Any, Callable

from psygnal import EmissionInfo, Signal, SignalInstance

from cellier.models.scene import DimsManager, DimsState
from cellier.types import DimsId

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DimsControlsUpdateEvent:
    """Event data that is emitted when the state of a dims controls is updated.

    Parameters
    ----------
    id : DimsId
        The ID of the dims model that the controls are for.
    state : dict[str, Any]
        The state of the dims model to update.
        The key is the string name of the parameters and
        the value is the value to set.
    controls_update_callback : Callable | None
        The callback function to block when the dims model is updated.
        This is the callback function that is called when the dims model is updated.
        This is used to prevent the update from bouncing back to the GUI.
    """

    id: DimsId
    state: dict[str, Any]
    controls_update_callback: Callable | None = None


class SceneEventBus:
    """Event bus for keeping the scene models in sync with the view.

    Events types:
        - dims: emitted when the dims model is updated
        - camera: emitted when the camera model is updated
        - rendered: emitted when the scene has completed a draw.
    """

    def __init__(
        self,
    ):
        # the signals for each dims model that has been registered
        self._dims_signals: dict[DimsId, SignalInstance] = {}

        # the signals for each dim control that has been registered
        self._dims_control_signals: dict[DimsId, SignalInstance] = {}

    @property
    def dims_signals(self) -> dict[DimsId, SignalInstance]:
        """Return the signals for each registered dims model.

        The dictionary key is the dims model ID and the value is the SignalInstance.
        """
        return self._dims_signals

    @property
    def dims_control_signals(self) -> dict[DimsId, SignalInstance]:
        """Return the signals for each registered dims control.

        The dictionary key is the dims model ID and the value is the SignalInstance.
        """
        return self._dims_control_signals

    def add_dims_with_controls(self, dims_model: DimsManager, dims_controls):
        """Add a dims model with a control UI to the event bus.

        This is a convenience method to register a dims model with a control UI.
        Generally, this is the method one should use to connect a dims model with
        a UI.

        Parameters
        ----------
        dims_model : DimsManager
            The dims model to register.
        dims_controls :
            The dims controls to register and connect to the dims model.
            The dims_controls must have the following:
                - a signal named currentIndexChanged that emits a
                  DimsControlsUpdateEvent when the dims controls are updated.
                - a method named _on_dims_state_changed that takes a DimsState object
                  and updates the dims controls to match the DimsState.
        """
        # register the dims model with the event bus
        self.register_dims(dims=dims_model)

        # register the dims controls with the event bus
        self.register_dims_controls(
            dims_id=dims_model.id,
            signal=dims_controls.currentIndexChanged,
        )

        # subscribe the dims model to the dims controls
        self.subscribe_to_dims_control(
            dims_id=dims_model.id,
            callback=dims_model.update_state,
        )

        # subscribe the dims controls to the dims model
        self.subscribe_to_dims(
            dims_id=dims_model.id,
            callback=dims_controls._on_dims_state_changed,
        )

    def register_dims(self, dims: DimsManager):
        """Register a DimsManager with the event bus.

        This will create a signal on the event bus that will be
        emitted when the dims model updates. Other components (e.g., GUI)
        can register to this signal to be notified of changes via the
        subscribe_to_dims() method.

        Parameters
        ----------
        dims : DimsManager
            The dims model to register.
        """
        if dims.id in self.dims_signals:
            logging.info(f"Dims {dims.id} is already registered.")
            return

        # connect the update event
        dims.events.all.connect(self._on_dims_model_update)

        # initialize the dims model callbacks
        self.dims_signals[dims.id] = SignalInstance(
            name=dims.id, check_nargs_on_connect=False, check_types_on_connect=False
        )

    def subscribe_to_dims(self, dims_id: DimsId, callback: Callable):
        """Subscribe to the dims model update signal.

        Parameters
        ----------
        dims_id : DimsId
            The ID of the dims model to subscribe to.
        callback : Callable
            The callback function to call when the dims model updates.
        """
        if dims_id not in self.dims_signals:
            raise ValueError(f"Dims {dims_id} is not registered.") from None

        # connect the callback to the signal
        self.dims_signals[dims_id].connect(callback)

    def register_dims_controls(self, dims_id: DimsId, signal: SignalInstance):
        """Register a signal for the dims controls.

        This creates a signal on the event bus that will be emitted when
        the dims control updates. Dims models can subscribe to this
        signal to be notified of changes.

        Parameters
        ----------
        dims_id : DimsId
            The ID of the dims model to register.
        signal : SignalInstance
            The signal instance on the dims control to register.
            This signal must emit a DimsState object.
        """
        if dims_id not in self.dims_control_signals:
            self.dims_control_signals[dims_id] = SignalInstance(
                name=dims_id,
                check_nargs_on_connect=False,
                check_types_on_connect=False,
            )

        # connect the callback to the signal
        signal.connect(self._on_dims_control_update)

    def subscribe_to_dims_control(self, dims_id: DimsId, callback: Callable):
        """Subscribe to the dims control update signal.

        Parameters
        ----------
        dims_id : DimsId
            The ID of the dims model to subscribe to.
        callback : Callable
            The callback function to call when the dims control updates.
        """
        if dims_id not in self.dims_control_signals:
            raise ValueError(f"Dims {dims_id} is not registered.") from None

        # connect the callback to the signal
        self.dims_control_signals[dims_id].connect(callback)

    def _on_dims_model_update(self, event: EmissionInfo):
        """Handle the update event for the dims model.

        This will emit the signal for the dims model.
        """
        # get the sender of the event
        dims: DimsManager = Signal.sender()

        dims_state: DimsState = dims.to_dims_state()

        # emit the signal for the dims model
        self.dims_signals[dims.id].emit(dims_state)

    def _on_dims_control_update(
        self,
        event: DimsControlsUpdateEvent,
    ):
        """Handle the update event for the dims control.

        This will emit the signal for the dims model.
        """
        # get the dims model ID the controls is for.
        dims_id = event.id

        try:
            control_signal = self.dims_control_signals[dims_id]
        except KeyError:
            logger.debug(
                "EventBus received event from control"
                f" for dims model {dims_id},"
                " but the model is not registered"
            )
            return

        callback_to_block = event.controls_update_callback

        if dims_id in self.dims_signals:
            # block the callback to prevent the update from bouncing back
            # to the GUI
            dims_signal = self.dims_signals[dims_id]

            # temporarily disconnect the callback and emit the event
            dims_signal.disconnect(callback_to_block, missing_ok=True)
            control_signal.emit(event.state)
            dims_signal.connect(callback_to_block)
        else:
            # if the dims model is not registered, just emit the event
            control_signal.emit(event.state)
