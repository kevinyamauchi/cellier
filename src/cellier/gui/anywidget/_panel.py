"""Composite anywidget control panel (Design A).

A single ``ControlPanel`` renders every control as raw DOM inside one ESM,
synced through traitlets.  Phase 1 implements the dims sliders only; later
phases add the appearance controls as additional traits + DOM rows (see the
design doc sections 6.3 / 6.4 and the Phase 2 list in the plan).

The panel satisfies the :class:`cellier.gui._protocol.WidgetView` contract: it
carries one ``_id``, a ``changed`` / ``closed`` psygnal pair, a ``widget``
property, and ``subscription_specs()`` -- exactly like the Qt widgets, so
``CellierController.connect_widget`` wires it the same way.

Notes on the traitlet <-> JSON boundary
----------------------------------------
JSON object keys are always strings, so the synced dict traits (``slice_indices``,
``axis_ranges``, ``axis_labels``) use ``str(axis_index)`` keys.  The cellier
layer converts to / from ``int`` axis indices at the event boundary
(``_on_dims_changed`` in, ``_emit_dims`` out).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import anywidget
import traitlets
from psygnal import Signal

from cellier.events import DimsChangedEvent, DimsUpdateEvent, SubscriptionSpec

if TYPE_CHECKING:
    from uuid import UUID

    from cellier.scene.scene import Scene

_STATIC = Path(__file__).parent / "static"


class ControlPanel(anywidget.AnyWidget):
    """Composite raw-DOM control panel for the anywidget front-end.

    Phase 1 exposes the per-axis dims sliders.  Construct via
    :meth:`from_scene`, then wire with::

        panel = ControlPanel.from_scene(scene, axis_ranges)
        controller.connect_widget(panel, subscription_specs=panel.subscription_specs())

    Parameters
    ----------
    scene_id : UUID
        UUID of the scene whose slice indices this panel controls.
    axis_ranges : dict
        Mapping of ``str(axis)`` to ``[min, max]`` slider bounds.
    axis_labels : dict
        Mapping of ``str(axis)`` to display label.
    slice_indices : dict
        Mapping of ``str(axis)`` to the current integer slice position.
    displayed_axes : list
        Axis indices currently displayed (their rows are hidden).
    stacked_axes : list
        Axis indices composited by the render layer (their rows are hidden).
    non_displayed : list
        Axis indices never shown as sliders regardless of dims state.
    """

    _esm = _STATIC / "panel.js"
    _css = _STATIC / "panel.css"

    # psygnal outward signals (the WidgetView contract); not traitlets.
    changed: Signal = Signal(object)
    closed: Signal = Signal()

    # Synced traits (one per field).  Dict keys are ``str(axis)``.
    slice_indices = traitlets.Dict().tag(sync=True)
    axis_labels = traitlets.Dict().tag(sync=True)
    axis_ranges = traitlets.Dict().tag(sync=True)
    displayed_axes = traitlets.List().tag(sync=True)
    stacked_axes = traitlets.List().tag(sync=True)
    non_displayed = traitlets.List().tag(sync=True)

    def __init__(
        self,
        *,
        scene_id: UUID,
        axis_ranges: dict,
        axis_labels: dict,
        slice_indices: dict,
        displayed_axes: list | tuple = (),
        stacked_axes: list | tuple = (),
        non_displayed: list | tuple = (),
        **kwargs,
    ) -> None:
        super().__init__(
            slice_indices={str(k): int(v) for k, v in slice_indices.items()},
            axis_labels={str(k): str(v) for k, v in axis_labels.items()},
            axis_ranges={
                str(k): [float(lo), float(hi)] for k, (lo, hi) in axis_ranges.items()
            },
            displayed_axes=[int(a) for a in displayed_axes],
            stacked_axes=[int(a) for a in stacked_axes],
            non_displayed=[int(a) for a in non_displayed],
            **kwargs,
        )
        self._id = uuid4()
        self._scene_id = scene_id
        # Guard around programmatic (model -> widget) trait writes so the
        # observe handler does not echo them back onto the bus.
        self._applying = False
        self.observe(self._on_slice_indices, names="slice_indices")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_scene(
        cls,
        scene: Scene,
        axis_ranges: dict[int, tuple[float, float]],
        *,
        non_displayed: tuple[int, ...] = (),
    ) -> ControlPanel:
        """Build a panel from a live scene, paralleling the Qt canvas widget.

        Derives axis labels and the initial dims state from *scene*; the caller
        supplies *axis_ranges* (which needs data-store knowledge not on the
        dims model itself).

        Parameters
        ----------
        scene : Scene
            The scene whose dims this panel controls.
        axis_ranges : dict[int, tuple[float, float]]
            Mapping of axis index to ``(world_min, world_max)``.
        non_displayed : tuple[int, ...]
            Axes excluded from the sliders regardless of dims state.

        Returns
        -------
        ControlPanel
        """
        axis_labels = dict(enumerate(scene.dims.coordinate_system.axis_labels))
        selection = scene.dims.selection
        return cls(
            scene_id=scene.id,
            axis_ranges=axis_ranges,
            axis_labels=axis_labels,
            slice_indices=dict(getattr(selection, "slice_indices", {})),
            displayed_axes=getattr(selection, "displayed_axes", ()),
            stacked_axes=getattr(selection, "stacked_axes", ()),
            non_displayed=non_displayed,
        )

    # ------------------------------------------------------------------
    # WidgetView contract
    # ------------------------------------------------------------------

    @property
    def widget(self) -> ControlPanel:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """Return the inbound subscription this panel requires."""
        return [
            SubscriptionSpec(
                event_type=DimsChangedEvent,
                handler=self._on_dims_changed,
                entity_id=self._scene_id,
            )
        ]

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    # ------------------------------------------------------------------
    # model -> widget
    # ------------------------------------------------------------------

    def _on_dims_changed(self, event: DimsChangedEvent) -> None:
        if event.source_id == self._id:
            return  # echo from our own slider move; ignore

        selection = event.dims_state.selection
        # Update slider values for the sliced axes.
        new_slices = dict(self.slice_indices)
        for axis, value in selection.slice_indices.items():
            new_slices[str(axis)] = int(value)
        self._set_field("slice_indices", new_slices)

        # Refresh which rows are hidden.
        self._set_field("displayed_axes", [int(a) for a in selection.displayed_axes])
        stacked = getattr(selection, "stacked_axes", ())
        self._set_field("stacked_axes", [int(a) for a in stacked])

    def _set_field(self, name: str, value) -> None:
        """Set a synced trait without echoing it back onto the bus."""
        self._applying = True
        try:
            setattr(self, name, value)
        finally:
            self._applying = False

    # ------------------------------------------------------------------
    # widget -> model
    # ------------------------------------------------------------------

    def _on_slice_indices(self, change) -> None:
        if self._applying:
            return
        self._emit_dims()

    def _emit_dims(self) -> None:
        """Emit a DimsUpdateEvent for the sliced (visible) axes only."""
        hidden = (
            set(self.displayed_axes) | set(self.stacked_axes) | set(self.non_displayed)
        )
        updates = {
            int(axis): int(value)
            for axis, value in self.slice_indices.items()
            if int(axis) not in hidden
        }
        self.changed.emit(
            DimsUpdateEvent(
                source_id=self._id,
                scene_id=self._scene_id,
                slice_indices=updates,
                displayed_axes=None,
            )
        )
