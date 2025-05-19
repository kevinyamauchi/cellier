from typing import Callable

import numpy as np
from psygnal import SignalInstance

from cellier.convenience import get_dims_with_canvas_id, get_dims_with_visual_id
from cellier.models.viewer import ViewerModel
from cellier.render._data_classes import (
    RendererCanvasMouseEvent,
    RendererVisualMouseEvent,
)
from cellier.types import (
    CanvasId,
    MouseCallbackData,
    VisualId,
)


class MouseEventBus:
    """A class to manage events for mouse interactions on the canvas.

    This is currently only implemented for the pygfx backend.
    """

    def __init__(self, viewer_model: ViewerModel | None = None):
        # store the viewer model, if provided
        self._viewer_model = viewer_model

        # instantiate the signals storage
        self._visual_signals: dict[VisualId, SignalInstance] = {}
        self._canvas_signals: dict[CanvasId, SignalInstance] = {}

    @property
    def visual_signals(self) -> dict[VisualId, SignalInstance]:
        """Returns the signal for each registered visual.

        The signal will emit a MouseCallback data object when
        a mouse interaction occurs.
        """
        return self._visual_signals

    @property
    def canvas_signals(self) -> dict[CanvasId, SignalInstance]:
        """Returns the signal for each registered canvas.

        The signal will emit a MouseCallback data object when
        a mouse interaction occurs.
        """
        return self._canvas_signals

    def register_visual(
        self,
        visual_id: VisualId,
    ) -> None:
        """Register a visual as a source for mouse events."""
        if visual_id not in self.visual_signals:
            self.visual_signals[visual_id] = SignalInstance(
                name=visual_id,
                check_nargs_on_connect=False,
                check_types_on_connect=False,
            )

    def register_canvas(self, canvas_id: CanvasId):
        """Register a canvas as a source for mouse events."""
        if canvas_id not in self.canvas_signals:
            self.canvas_signals[canvas_id] = SignalInstance(
                name=canvas_id,
                check_nargs_on_connect=False,
                check_types_on_connect=False,
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

    def subscribe_to_canvas(self, canvas_id: CanvasId, callback: Callable):
        """Subscribe to mouse events for a canvas.

        Parameters
        ----------
        canvas_id : CanvasId
            The ID of the canvas to subscribe to.
        callback : Callable
            The callback to call when a mouse event occurs.
        """
        if canvas_id not in self.canvas_signals:
            raise ValueError(f"Canvas {canvas_id} is not registered.") from None

        # connect the callback to the signal
        self.canvas_signals[canvas_id].connect(callback)

    def _on_mouse_event(
        self, event: RendererVisualMouseEvent | RendererCanvasMouseEvent
    ):
        """Receive and re-emit the mouse event.

        This method:
            1. Receives the mouse callback from the rendering backend (PyGfx)
            2. Formats the event into a MouseCallbackData object
            3. Calls all registered callbacks via the Signal

        Currently, this only handles pygfx mouse events.
        In the future, we could extend this for other backends.

        Parameters
        ----------
        event : PygfxPointerEvent
            The mouse event from the rendering backend.
            Currently, this must be a pygfx PointerEvent.
        """
        if isinstance(event, RendererCanvasMouseEvent):
            # get the dims manager to determine the click coordinate
            dims_manager = get_dims_with_canvas_id(
                viewer_model=self._viewer_model,
                canvas_id=event.source_id,
            )
        elif isinstance(event, RendererVisualMouseEvent):
            # get the dims manager to determine the click coordinate
            dims_manager = get_dims_with_visual_id(
                viewer_model=self._viewer_model,
                visual_id=event.source_id,
            )
        else:
            raise TypeError(f"Unknown event type: {type(event)}")

        n_displayed_dims = dims_manager.selection.n_displayed_dims
        displayed_world_coordinate = event.coordinate[:n_displayed_dims]

        # determine the coordinate of the click
        coordinate = np.asarray(dims_manager.selection.index_selection)
        displayed_dimensions = list(dims_manager.selection.ordered_dims)[
            -n_displayed_dims:
        ]
        for pick_coordinate, dims_index in zip(
            displayed_world_coordinate, displayed_dimensions
        ):
            coordinate[dims_index] = pick_coordinate

        # make the data
        mouse_event_data = MouseCallbackData(
            visual_id=event.source_id,
            type=event.type,
            button=event.button,
            modifiers=event.modifiers,
            coordinate=coordinate,
            pick_info=event.pick_info,
        )

        if isinstance(event, RendererVisualMouseEvent):
            # get the signal for the visual
            try:
                signal = self.visual_signals[event.source_id]
            except KeyError:
                # if the visual is not registered, we can ignore the event
                return
        elif isinstance(event, RendererCanvasMouseEvent):
            try:
                signal = self.canvas_signals[event.source_id]
            except KeyError:
                # if the canvas is not registered, we can ignore the event
                return
        else:
            raise TypeError(f"Unknown event type: {type(event)}")

        # emit the callback
        signal.emit(mouse_event_data)
