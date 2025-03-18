from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Callable

from psygnal import SignalInstance

from cellier.types import VisualId

if TYPE_CHECKING:
    from pygfx import PointerEvent as PygfxPointerEvent


class MouseButton(Enum):
    """Mouse buttons for mouse click events."""

    NONE = "none"
    LEFT = "left"
    MIDDLE = "middle"
    RIGHT = "right"


class MouseModifiers(Enum):
    """Keyboard modifiers for mouse click events."""

    SHIFT = "shift"
    CTRL = "ctrl"
    ALT = "alt"
    META = "meta"


# convert the pygfx mouse buttons to the Cellier mouse buttons
# https://jupyter-rfb.readthedocs.io/en/stable/events.html
pygfx_buttons_to_cellier_buttons = {
    0: MouseButton.NONE,
    1: MouseButton.LEFT,
    2: MouseButton.RIGHT,
    3: MouseButton.MIDDLE,
}

# convert the pygfx modifiers to the Cellier modifiers
# https://jupyter-rfb.readthedocs.io/en/stable/events.html
pygfx_modifiers_to_cellier_modifiers = {
    "Shift": MouseModifiers.SHIFT,
    "Control": MouseModifiers.CTRL,
    "Alt": MouseModifiers.ALT,
    "Meta": MouseModifiers.META,
}


@dataclass(frozen=True)
class MouseCallbackData:
    """Data from a mouse click on the canvas."""

    visual_id: VisualId
    button: MouseButton
    modifiers: list[MouseModifiers]
    pick_info: dict[str, Any]


class MouseEventBus:
    """A class to manage events for mouse interactions on the canvas.

    This is currently only implemented for the pygfx backend.
    """

    def __init__(self):
        self._visual_signals: dict[VisualId, SignalInstance] = {}

    @property
    def visual_signals(self) -> dict[VisualId, SignalInstance]:
        """Returns the signal for each registered visual.

        The signal will emit a MouseCallback data object when
        a mouse interaction occurs.
        """
        return self._visual_signals

    def register_visual(
        self, visual_id: VisualId, callback_handlers: list[Callable]
    ) -> None:
        """Register a visual as a source for mouse events.

        This assumes a pygfx backend. In the future, we could consider
        allowing for other backends
        """
        if visual_id not in self.visual_signals:
            self.visual_signals[visual_id] = SignalInstance(
                name=visual_id,
                check_nargs_on_connect=False,
                check_types_on_connect=False,
            )

        # connect all events to the visual model update handler
        for handler in callback_handlers:
            handler(
                partial(self._on_mouse_event, visual_id=visual_id),
                *("pointer_down", "pointer_up", "pointer_move"),
            )

    def subscribe_to_visual(self, visual_id: VisualId, callback: Callable):
        """Subscribe to mouse events for a visual.

        This will call the callback when a mouse event occurs on the
        visual.

        Parameters
        ----------
        visual_id : VisualId
            The ID of the visual to subscribe to.
        callback : Callable
            The callback to call when a mouse event occurs.
            This must accept a MouseCallbackData object as the only argument.
        """
        if visual_id not in self.visual_signals:
            raise ValueError(f"Visual {visual_id} is not registered.") from None

        # connect the callback to the signal
        self.visual_signals[visual_id].connect(callback)

    def _on_mouse_event(self, event: "PygfxPointerEvent", visual_id: VisualId):
        """Receive and re-emit the mouse event.

        This method:
            1. Receives the mouse callback from the rendering backend (PyGfx)
            2. Formats the event into a MouseCallbackData object
            3. Calls all registered

        Currently, this only handles pygfx mouse events.
        In the future, we could extend this for other backends.

        Parameters
        ----------
        event : PygfxPointerEvent
            The mouse event from the rendering backend.
            Currently, this must be a pygfx PointerEvent.
        visual_id : VisualId
            The ID for the visual the event is associated with.
        """
        # get the mouse button
        button = pygfx_buttons_to_cellier_buttons[event.button]

        # get the modifiers
        modifiers = [
            pygfx_modifiers_to_cellier_modifiers[key] for key in event.modifiers
        ]

        # make the data
        mouse_event_data = MouseCallbackData(
            visual_id=visual_id,
            button=button,
            modifiers=modifiers,
            pick_info=event.pick_info,
        )

        # get the signal for the visual
        signal = self.visual_signals[visual_id]

        # emit the callback
        signal.emit(mouse_event_data)
