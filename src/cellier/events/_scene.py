import logging
from typing import Callable

from psygnal import EmissionInfo, Signal, SignalInstance

from cellier.models.scene import (
    DimsManager,
    DimsState,
    OrthographicCamera,
    PerspectiveCamera,
)
from cellier.types import (
    CameraControlsUpdateEvent,
    CameraId,
    DimsControlsUpdateEvent,
    DimsId,
)

logger = logging.getLogger(__name__)


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

        # the signals for each camera model that has been registered
        self._camera_signals: dict[CameraId, SignalInstance] = {}

        # the signals for each camera control that has been registered
        self._camera_controls: dict[CameraId, SignalInstance] = {}

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

    @property
    def camera_signals(self) -> dict[CameraId, SignalInstance]:
        """Return the signals for each registered camera model.

        The dictionary key is the camera model ID and the value is the SignalInstance.
        """
        return self._camera_signals

    @property
    def camera_controls(self) -> dict[CameraId, SignalInstance]:
        """Return the signals for each registered camera control.

        The dictionary key is the camera model ID and the value is the SignalInstance.
        """
        return self._camera_controls

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

    def register_camera(self, camera_model: PerspectiveCamera | OrthographicCamera):
        """Register a camera model with the event bus.

        This will create a signal on the event bus that will be emitted
        when the camera model updates. Other components (e.g., GUI) can
        register to this signal to be notified of changes via the
        subscribe_to_camera() method.

        Parameters
        ----------
        camera_model : PerspectiveCamera | OrthographicCamera
            The camera model to register.
        """
        if camera_model.id in self.camera_signals:
            logging.info(f"Camera {camera_model.id} is already registered.")
            return

        # connect the update event
        camera_model.events.all.connect(self._on_camera_model_update)

        # initialize the camera model callbacks
        self.camera_signals[camera_model.id] = SignalInstance(
            name=camera_model.id,
            check_nargs_on_connect=False,
            check_types_on_connect=False,
        )

    def subscribe_to_camera(self, camera_id: CameraId, callback: Callable):
        """Subscribe to the camera model update signal.

        Parameters
        ----------
        camera_id : CameraId
            The ID of the camera model to subscribe to.
        callback : Callable
            The callback function to call when the camera model updates.
        """
        if camera_id not in self.camera_signals:
            raise ValueError(f"Camera {camera_id} is not registered.") from None

        # connect the callback to the signal
        self.camera_signals[camera_id].connect(callback)

    def register_camera_controls(self, camera_id: CameraId, signal: SignalInstance):
        """Register a controller that can update the camera model.

        This will create a signal on the event bus that will be emitted
        when the camera control updates. The camera model can subscribe to this
        signal to be notified of changes.

        Parameters
        ----------
        camera_id : CameraId
            The ID of the camera model to register.
        signal : SignalInstance
            The signal instance on the camera control that emits when
            the controls are updated.
        """
        if camera_id not in self.camera_controls:
            # create the signal that camera models can subscribe to
            self.camera_controls[camera_id] = SignalInstance(
                name=camera_id,
                check_nargs_on_connect=False,
                check_types_on_connect=False,
            )

        # connect the callback to the signal
        signal.connect(self._on_camera_control_update)

    def subscribe_to_camera_controls(self, camera_id: CameraId, callback: Callable):
        """Subscribe a camera model to the event for when the controls are updated.

        Parameters
        ----------
        camera_id : CameraId
            The ID of the camera model to subscribe to.
        callback : Callable
            The callback function to call when the camera control updates.
        """
        try:
            camera_control_signal = self.camera_controls[camera_id]
        except KeyError:
            raise ValueError(f"Camera {camera_id} is not registered.") from None

        # connect the callback to the signal
        camera_control_signal.connect(callback)

    def _on_dims_model_update(self, event: EmissionInfo):
        """Handle the update event for the dims model.

        This will emit the signal for the dims model.
        """
        # get the sender of the event
        dims: DimsManager = Signal.sender()

        dims_state: DimsState = dims.to_state()

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

    def _on_camera_model_update(self, event: EmissionInfo):
        # get the sender
        camera_model = Signal.sender()

        try:
            signal = self.camera_signals[camera_model.id]
        except KeyError:
            logger.debug(
                f"EventBus received event from camera model {camera_model.id},"
                "but the model is not registered"
            )

        # dictionary with the updated state
        new_state = camera_model.to_state()
        signal.emit(new_state)

    def _on_camera_control_update(self, event: CameraControlsUpdateEvent):
        """Handle the update event for the camera control.

        This will emit the signal for the camera model.
        """
        # get the sender
        camera_id = event.id

        try:
            signal = self.camera_controls[camera_id]
        except KeyError:
            logger.debug(
                f"EventBus received event from camera control {camera_id},"
                "but the model is not registered"
            )
            return

        callback_to_block = event.controls_update_callback

        # update the camera model
        if camera_id in self.camera_signals:
            # block the callback to prevent the update from bouncing back
            # to the GUI
            camera_signal = self.camera_signals[camera_id]

            # temporarily disconnect the callback and emit the event
            camera_signal.disconnect(callback_to_block, missing_ok=True)
            signal.emit(event.state)
            camera_signal.connect(callback_to_block)
        else:
            # if the camera model is not registered, just emit the event
            signal.emit(event.state)
