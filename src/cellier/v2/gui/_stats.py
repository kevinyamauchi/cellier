"""Qt performance statistics widget for cellier v2."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from cellier.v2.controller import CellierController
    from cellier.v2.stats import FetchStats, PerformanceStats

_ALL_VISUALS = "all visuals"
_NO_DATA = "—"
# Readings older than this many seconds are shown greyed-out.
_STALE_THRESHOLD_S = 5.0


def _age_label(last_updated: float) -> str:
    """Human-readable age string for a stats timestamp."""
    if last_updated == 0.0:
        return " (no data)"
    age = time.perf_counter() - last_updated
    if age < 2.0:
        return ""
    return f" ({age:.0f}s ago)"


class QtPerformanceWidget:
    """Collapsible Qt widget displaying live render and fetch performance stats.

    Shows per-canvas FPS and draw time (updated via :class:`FrameRenderedEvent`)
    and per-visual fetch timing (polled every 300 ms via a ``QTimer``).  A
    combo-box lets the user select which visual's fetch stats to display; the
    list is kept in sync with :class:`VisualAddedEvent` /
    :class:`VisualRemovedEvent` events.

    Do not instantiate directly — use :meth:`from_controller`.

    Parameters
    ----------
    stats :
        The shared :class:`~cellier.v2.stats.PerformanceStats` instance
        from the controller.
    visual_names_fn :
        Called with no arguments; returns a ``{visual_uuid: name}`` mapping
        reflecting the current state of the viewer model.
    canvas_ids :
        Ordered list of canvas UUIDs to display render stats for.
    owner_id :
        UUID used to register event-bus subscriptions; must be passed to
        ``controller._event_bus.unsubscribe_all`` on teardown.
    parent :
        Optional Qt parent widget.
    """

    def __init__(
        self,
        stats: PerformanceStats,
        visual_names_fn,
        canvas_ids: list[UUID],
        owner_id: UUID,
        *,
        parent=None,
    ) -> None:
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import (
            QCheckBox,
            QComboBox,
            QFormLayout,
            QLabel,
            QWidget,
        )
        from superqt import QCollapsible

        self._stats = stats
        self._visual_names_fn = visual_names_fn
        self._canvas_ids = canvas_ids
        self._owner_id = owner_id
        # Set by from_controller; updated on each enable/disable cycle.
        self._frame_handle = None
        self._controller_ref = None

        # Last known {uuid: name} used to detect changes and rebuild combobox.
        self._known_visuals: dict[UUID, str] = {}

        self._collapsible = QCollapsible("performance", parent=parent)
        content = QWidget()
        form = QFormLayout(content)
        form.setContentsMargins(4, 4, 4, 4)
        self._collapsible.addWidget(content)

        # ── Enable/disable toggle ────────────────────────────────────────
        self._enable_checkbox = QCheckBox("Enable monitoring")
        self._enable_checkbox.setChecked(True)
        self._enable_checkbox.toggled.connect(self._on_monitoring_toggled)
        form.addRow(self._enable_checkbox)

        # ── Render section ───────────────────────────────────────────────
        self._render_labels: dict[UUID, tuple[QLabel, QLabel]] = {}
        for canvas_id in canvas_ids:
            fps_label = QLabel(_NO_DATA)
            draw_label = QLabel(_NO_DATA)
            suffix = f" [{canvas_id}]" if len(canvas_ids) > 1 else ""
            form.addRow(f"FPS{suffix}", fps_label)
            form.addRow(f"draw time{suffix}", draw_label)
            self._render_labels[canvas_id] = (fps_label, draw_label)

        # ── Fetch section ────────────────────────────────────────────────
        self._fetch_combo = QComboBox()
        self._fetch_combo.addItem(_ALL_VISUALS)
        form.addRow("visual", self._fetch_combo)

        self._fetch_chunks_label = QLabel(_NO_DATA)
        self._fetch_ms_per_chunk_label = QLabel(_NO_DATA)
        form.addRow("chunks", self._fetch_chunks_label)
        form.addRow("ms / chunk", self._fetch_ms_per_chunk_label)

        # ── Timer for fetch stats refresh ────────────────────────────────
        self._timer = QTimer()
        self._timer.setInterval(300)
        self._timer.timeout.connect(self._refresh_fetch)
        self._timer.start()

    # ── Class method constructor ─────────────────────────────────────────

    @classmethod
    def from_controller(
        cls,
        controller: CellierController,
        *,
        parent=None,
    ) -> QtPerformanceWidget:
        """Construct a widget wired to *controller*'s live stats.

        Subscribes to :class:`FrameRenderedEvent`, :class:`VisualAddedEvent`,
        and :class:`VisualRemovedEvent` on the controller's event bus.
        Call :meth:`close` to unsubscribe when the widget is destroyed.

        Parameters
        ----------
        controller :
            The running :class:`~cellier.v2.controller.CellierController`.
        parent :
            Optional Qt parent widget.
        """
        owner_id = uuid4()
        canvas_ids = list(controller._render_manager._canvases.keys())

        def visual_names_fn() -> dict[UUID, str]:
            result: dict[UUID, str] = {}
            for scene in controller._model.scenes.values():
                for visual in scene.visuals:
                    result[visual.id] = visual.name
            return result

        widget = cls(
            stats=controller.stats,
            visual_names_fn=visual_names_fn,
            canvas_ids=canvas_ids,
            owner_id=owner_id,
            parent=parent,
        )

        # Store before subscribing so _on_monitoring_toggled can use it.
        widget._controller_ref = controller

        widget._frame_handle = controller.on_frame_rendered(
            widget._on_frame_rendered,
            owner_id=owner_id,
        )
        controller.on_any_visual_added(
            widget._on_visual_list_changed,
            owner_id=owner_id,
        )
        controller.on_any_visual_removed(
            widget._on_visual_list_changed,
            owner_id=owner_id,
        )

        return widget

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def widget(self):
        """The ``QCollapsible`` to insert into a Qt layout."""
        return self._collapsible

    def close(self) -> None:
        """Unsubscribe from all event-bus callbacks and stop the timer.

        Call this when the widget is being destroyed or hidden permanently.
        Safe to call whether monitoring is currently enabled or disabled.
        """
        self._timer.stop()
        controller = self._controller_ref
        if controller is not None:
            controller._event_bus.unsubscribe_all(self._owner_id)
            self._frame_handle = None

    # ── Event callbacks (run on Qt main thread) ──────────────────────────

    def _on_monitoring_toggled(self, enabled: bool) -> None:
        controller = self._controller_ref
        if controller is None:
            return
        if enabled:
            self._frame_handle = controller.on_frame_rendered(
                self._on_frame_rendered,
                owner_id=self._owner_id,
            )
            self._timer.start()
        else:
            if self._frame_handle is not None:
                controller._event_bus.unsubscribe(self._frame_handle)
                self._frame_handle = None
            self._timer.stop()
            self._clear_labels()

    def _on_frame_rendered(self, event) -> None:
        canvas_id = event.canvas_id
        labels = self._render_labels.get(canvas_id)
        if labels is None:
            return
        render_stats = self._stats.render.get(canvas_id)
        if render_stats is None:
            return
        fps_label, draw_label = labels
        fps_label.setText(f"{render_stats.fps:.1f}")
        draw_label.setText(f"{render_stats.draw_time_ms:.1f} ms")

    def _on_visual_list_changed(self, event) -> None:
        """Rebuild the fetch combo-box when any visual is added or removed."""
        self._sync_visual_combobox()

    # ── Internal helpers ─────────────────────────────────────────────────

    def _clear_labels(self) -> None:
        """Reset all value labels to the no-data sentinel."""
        for fps_label, draw_label in self._render_labels.values():
            fps_label.setText(_NO_DATA)
            draw_label.setText(_NO_DATA)
        for lbl in (
            self._fetch_chunks_label,
            self._fetch_ms_per_chunk_label,
        ):
            lbl.setText(_NO_DATA)

    def _sync_visual_combobox(self) -> None:
        """Rebuild the combobox if the visual list has changed."""
        current = self._visual_names_fn()
        if current == self._known_visuals:
            return

        self._known_visuals = dict(current)
        combo = self._fetch_combo

        # Preserve selection by name across the rebuild.
        selected_name = combo.currentText()

        combo.blockSignals(True)
        combo.clear()
        combo.addItem(_ALL_VISUALS)
        for name in sorted(current.values()):
            combo.addItem(name)
        combo.blockSignals(False)

        # Restore previous selection, falling back to "all visuals".
        idx = combo.findText(selected_name)
        combo.setCurrentIndex(max(0, idx))

    def _selected_fetch_stats(self) -> FetchStats | None:
        """Return the FetchStats for the currently selected combo entry."""
        name = self._fetch_combo.currentText()
        if name == _ALL_VISUALS:
            # Return the entry with the highest total_ms (worst offender).
            if not self._stats.fetch:
                return None
            return max(self._stats.fetch.values(), key=lambda s: s.total_ms)
        # Reverse-look up UUID from current known_visuals.
        for uid, n in self._known_visuals.items():
            if n == name:
                return self._stats.fetch.get(uid)
        return None

    def _refresh_fetch(self) -> None:
        """Poll fetch stats and update labels. Called by QTimer."""
        # Also sync the combobox here in case events were missed.
        self._sync_visual_combobox()

        fs = self._selected_fetch_stats()
        if fs is None or fs.last_updated == 0.0:
            for lbl in (
                self._fetch_chunks_label,
                self._fetch_ms_per_chunk_label,
            ):
                lbl.setText(_NO_DATA)
            return

        age = _age_label(fs.last_updated)
        cancelled = " (cancelled)" if fs.cancelled else ""
        self._fetch_chunks_label.setText(f"{fs.n_chunks}{cancelled}{age}")
        self._fetch_ms_per_chunk_label.setText(f"{fs.ms_per_chunk:.2f} ms")
