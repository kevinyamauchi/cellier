"""Dims-only anywidget -- one slice slider per non-displayed axis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import anywidget
import traitlets
from psygnal import Signal

from cellier.events import (
    DimsChangedEvent,
    DimsUpdateEvent,
    SubscriptionSpec,
)

if TYPE_CHECKING:
    from uuid import UUID

    from cellier.scene.scene import Scene

_STATIC = Path(__file__).parent / "static"


class AnywidgetDimsPanel(anywidget.AnyWidget):
    """Renders per-axis slice sliders for the anywidget front-end.

    Satisfies the :class:`cellier.gui._protocol.WidgetView` contract so
    ``CellierController.connect_widget`` wires it the same way as Qt widgets.

    Construct via :meth:`from_scene`, then wire with::

        dims = AnywidgetDimsPanel.from_scene(scene, axis_ranges)
        controller.connect_widget(dims, subscription_specs=dims.subscription_specs())
    """

    _esm = _STATIC / "dims_panel.js"
    _css = _STATIC / "dims_panel.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

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
        self._applying = False

        self.observe(self._on_slice_indices, names="slice_indices")

    @classmethod
    def from_scene(
        cls,
        scene: Scene,
        axis_ranges: dict[int, tuple[float, float]],
        *,
        non_displayed: tuple[int, ...] = (),
    ) -> AnywidgetDimsPanel:
        """Build a dims panel from a live scene."""
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

    @property
    def widget(self) -> AnywidgetDimsPanel:
        return self

    def subscription_specs(self) -> list[SubscriptionSpec]:
        return [
            SubscriptionSpec(
                event_type=DimsChangedEvent,
                handler=self._on_dims_changed,
                entity_id=self._scene_id,
            )
        ]

    def close(self) -> None:
        self.closed.emit()

    # ------------------------------------------------------------------
    # model -> widget
    # ------------------------------------------------------------------

    def _on_dims_changed(self, event: DimsChangedEvent) -> None:
        if event.source_id == self._id:
            return
        selection = event.dims_state.selection
        new_slices = dict(self.slice_indices)
        for axis, value in selection.slice_indices.items():
            new_slices[str(axis)] = int(value)
        self._set_field("slice_indices", new_slices)
        self._set_field("displayed_axes", [int(a) for a in selection.displayed_axes])
        stacked = getattr(selection, "stacked_axes", ())
        self._set_field("stacked_axes", [int(a) for a in stacked])

    def _set_field(self, name: str, value) -> None:
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
