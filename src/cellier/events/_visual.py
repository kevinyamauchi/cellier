"""Class to manage connecting events between the visual model and the view."""

import logging
from typing import Any, Callable

from psygnal import EmissionInfo, Signal, SignalInstance

from cellier.types import VisualType

logger = logging.getLogger(__name__)


class VisualEventBus:
    """Class to manage connecting events for the visual models.

    There are three types of events:
        - visual: communicate changes to the visual model state.
        - visual_controls: communicate changes to the visual gui state.
    """

    def __init__(self):
        # the signals for each visual model that has been registered
        self._visual_model_signals: dict[str, SignalInstance] = {}

        # the signals for each visual control that has been registered
        self._visual_control_signals: dict[str, SignalInstance] = {}

    @property
    def model_signals(self) -> dict[str, SignalInstance]:
        """Return the signals for each registered visual model.

        The dictionary key is the visual model ID and the value is the SignalInstance.
        """
        return self._visual_model_signals

    @property
    def control_signals(self) -> dict[str, SignalInstance]:
        """Return the signals for each registered visual control.

        The dictionary key is the visual model ID and the value is
        the SignalInstance.
        """
        return self._visual_control_signals

    def register_visual(self, visual: VisualType):
        """Register a visual with the event bus.

        This will create a signal on the event bus that will be
        emitted when the visual model updates. Other components (e.g., GUI)
        can register to this signal to be notified of changes via the
        subscribe_to_visual() method.

        Parameters
        ----------
        visual : VisualType
            The visual model to register.
        """
        if visual.id in self.model_signals:
            logging.info(f"Visual {visual.id} is already registered.")
            return
        # connect all events to the visual model update handler
        visual.events.all.connect(self._on_visual_model_update)

        # initialize the visual model callbacks
        self.model_signals[visual.id] = SignalInstance(
            name=visual.id, check_nargs_on_connect=False, check_types_on_connect=False
        )

    def subscribe_to_visual(self, visual_id: str, callback: Callable):
        """Subscribe to an event on a visual model.

        Parameters
        ----------
        visual_id : str
            The ID of the visual model to subscribe to.
        callback : Callable
            The callback to call when the visual model updates.
        """
        try:
            visual_model_signal = self.model_signals[visual_id]
        except KeyError:
            raise ValueError(f"Visual {visual_id} is not registered.") from None

        # connect the visual model signal
        visual_model_signal.connect(callback)

    def register_controls(self, visual_id: str, signal: SignalInstance):
        """Register a visual control with the event bus.

        This creates a signal on the event bus that will be emitted when
        the visual control updates. Visual models can subscribe to this
        signal to be notified of changes.

        Parameters
        ----------
        visual_id : str
            The ID of the visual model to register the control for.
        signal : SignalInstance
            The signal to register.
        """
        if visual_id not in self.control_signals:
            # If we haven't registered this visual_id yet, create the signal
            self.control_signals[visual_id] = SignalInstance(
                name=visual_id,
                check_nargs_on_connect=False,
                check_types_on_connect=False,
            )
        # connect the visual control signal
        signal.connect(self._on_visual_control_update)

    def subscribe_to_controls(self, visual_id: str, callback: Callable):
        """Subscribe to the event emitted when the visual controls are updated."""
        try:
            visual_control_signal = self.control_signals[visual_id]
        except KeyError:
            raise ValueError(f"Visual {visual_id} is not registered.") from None

        # connect the visual control signal
        visual_control_signal.connect(callback)

    def _on_visual_model_update(self, event: EmissionInfo):
        """Handle a visual model update event.

        This emits a dictionary containing the visual id and
        the updated property values.
        """
        # get the sender
        emitter_object = Signal.sender()

        try:
            signal = self.model_signals[emitter_object.id]
        except KeyError:
            logger.debug(
                f"EventBus received event from visual model {emitter_object.id},"
                "but the model is not registered"
            )

        # dictionary with the updated state
        property_name = event.signal.name
        property_value = event.args[0]
        new_state = {"id": emitter_object.id, property_name: property_value}
        signal.emit(new_state)

    def _on_visual_control_update(self, event: Any | None = None):
        """Handle a visual control update event.

        The update dictionary must include the following keys:
        - id: the id of the visual model
        - controls_update_callback: the visual control's update callback that
                                    should be blocked when updating the visual model.
                                    this prevents the update from bouncing back
                                    to the GUI.
        """
        visual_id = event["id"]
        try:
            control_signal = self.control_signals[visual_id]
        except KeyError:
            logger.debug(
                "EventBus received event from control"
                f" for visual model {visual_id},"
                " but the model is not registered"
            )
            return

        callback_to_block = event.pop("controls_update_callback")

        # update the visual model
        if visual_id in self.model_signals:
            # block the gui update callback so the signal doesn't bounce back
            visual_signal = self.model_signals[visual_id]

            # emit the event and temporarily disconnect the callback
            visual_signal.disconnect(callback_to_block, missing_ok=True)
            control_signal.emit(event)
            visual_signal.connect(callback_to_block)
        else:
            control_signal.emit(event)
