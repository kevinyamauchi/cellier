"""Widgets for scene components.

The QtDimsSliders and QtCanvasWidget classes are modified from
the _QDimsSliders and _QArrayViewer classes from ndv, respectively.
https://github.com/pyapp-kit/ndv/blob/main/src/ndv/views/_qt/_array_view.py

NDV license:
BSD 3-Clause License

Copyright (c) 2023, Talley Lambert

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

from uuid import uuid4

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QFormLayout, QSizePolicy, QVBoxLayout, QWidget
from superqt import QLabeledSlider

from cellier.v2.events import DimsUpdateEvent

SLIDER_STYLE = """
QSlider::groove:horizontal {
    border: 1px solid #bbb;
    background: white;
    height: 10px;
    border-radius: 4px;
}

QSlider::handle:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #eee, stop:1 #ccc);
    border: 1px solid #777;
    width: 13px;
    margin-top: -7px;
    margin-bottom: -7px;
    border-radius: 4px;
}

QSlider::add-page:horizontal {
    background: #fff;
    border: 1px solid #777;
    height: 10px;
    border-radius: 4px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1: 0, y1: 0,    x2: 0, y2: 1,
        stop: 0 #66e, stop: 1 #bbf);
    background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1,
        stop: 0 #bbf, stop: 1 #55f);
    border: 1px solid #777;
    height: 10px;
    border-radius: 4px;
}

QSlider::handle:horizontal:hover {
background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
    stop:0 #fff, stop:1 #ddd);
border: 1px solid #444;
border-radius: 4px;
}

QLabel { font-size: 12px; }
"""


class QtDimsSliders:
    """Bidirectional multi-axis dims slider panel wired to the cellier v2 bus.

    Composes a ``QWidget`` container (with a ``QFormLayout``) holding one
    ``QLabeledSlider`` per axis.  Sliders for displayed axes are hidden; only
    sliced (non-displayed) axes are shown.

    Follows the v2 widget pattern:

    - One ``UUID`` (``self._id``) per panel — all sliders share it.
    - Subscribed to ``DimsChangedEvent`` via ``controller.on_dims_changed``.
    - Echo-filters its own changes using ``source_id``.
    - Suppresses re-entrant slider signals with ``blockSignals`` when applying
      model-driven updates.

    Parameters
    ----------
    controller :
        The ``CellierController`` instance.
    scene_id :
        UUID of the scene whose slice indices this widget controls.
    axis_ranges :
        Mapping of axis index to ``(min, max)`` for each slider.
    axis_labels :
        Mapping of axis index to display label, e.g. ``{0: "z", 1: "y", 2: "x"}``.
    initial_slice_indices :
        Starting slider values; typically ``scene.dims.selection.slice_indices``.
    initial_displayed_axes :
        Axes to hide initially; typically ``scene.dims.selection.displayed_axes``.
    parent :
        Optional Qt parent widget for the internal container.
    """

    def __init__(
        self,
        controller,
        scene_id,
        axis_ranges: dict[int, tuple[int, int]],
        axis_labels: dict[int, str],
        *,
        initial_slice_indices: dict[int, int] | None = None,
        initial_displayed_axes: tuple[int, ...] = (),
        debounce_ms: int = 50,
        parent: QWidget | None = None,
    ) -> None:
        # ── Cellier layer ────────────────────────────────────────────────────
        self._id = uuid4()
        self._controller = controller
        self._scene_id = scene_id

        # Single-shot QTimer for rate-limiting rapid slider moves.
        # Fires at most once per interval; a dirty flag ensures the final
        # position is always submitted even if it landed between ticks.
        self._rate_limit_timer = QTimer()
        self._rate_limit_timer.setSingleShot(True)
        self._rate_limit_timer.setInterval(debounce_ms)
        self._rate_limit_timer.timeout.connect(self._on_rate_limit_tick)
        self._slider_dirty = False

        # ── Qt seam 1: build container and sliders ───────────────────────────
        self._container = QWidget(parent)
        self._container.setStyleSheet(SLIDER_STYLE)

        layout = QFormLayout(self._container)
        layout.setSpacing(2)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.setContentsMargins(0, 0, 0, 0)

        self._sliders: dict[int, QLabeledSlider] = {}
        self._displayed_axes: tuple[int, ...] = initial_displayed_axes
        _initial = initial_slice_indices or {}

        for axis, (min_val, max_val) in axis_ranges.items():
            sld = QLabeledSlider(Qt.Orientation.Horizontal)
            sld.setRange(min_val, max_val)
            sld.setValue(_initial.get(axis, min_val))
            # Capture `axis` by value in the default-argument closure.
            sld.valueChanged.connect(
                lambda value, ax=axis: self._on_slider_changed(ax, value)
            )
            layout.addRow(axis_labels.get(axis, str(axis)), sld)
            self._sliders[axis] = sld

        self._update_visibility(initial_displayed_axes)

        # Subscribe; owner_id groups all our subscriptions for bulk cleanup.
        controller.on_dims_changed(scene_id, self._on_dims_changed, owner_id=self._id)

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self) -> QWidget:
        """The Qt widget to insert into a layout.

        Qt seam 1: replace with the backend element for other toolkits.
        """
        return self._container

    def current_index(self) -> dict[int, int]:
        """Return the current value of every slider regardless of visibility."""
        return {axis: sld.value() for axis, sld in self._sliders.items()}

    def close(self) -> None:
        """Unsubscribe from the bus.  Call when the owning window closes."""
        self._controller.unsubscribe_owner(self._id)

    # ── Cellier layer: model → widget ────────────────────────────────────────

    def _on_dims_changed(self, event) -> None:
        if event.source_id == self._id:
            return  # echo from our own slider move; ignore

        # Update slider values for sliced axes.
        for axis in self._sliders:
            value = event.dims_state.selection.slice_indices.get(axis)
            if value is not None:
                self._set_value(axis, value)

        # Show/hide sliders based on which axes are now displayed.
        self._update_visibility(event.dims_state.selection.displayed_axes)

    # ── Cellier layer: widget → model ────────────────────────────────────────

    def _on_slider_changed(self, axis: int, value: int) -> None:
        self._slider_dirty = True
        if not self._rate_limit_timer.isActive():
            self._submit_slider_values()
            self._rate_limit_timer.start()

    def _on_rate_limit_tick(self) -> None:
        if self._slider_dirty:
            self._submit_slider_values()
            self._rate_limit_timer.start()

    def _submit_slider_values(self) -> None:
        """Submit current slider values for all sliced (non-displayed) axes."""
        self._slider_dirty = False
        updates = {
            axis: sld.value()
            for axis, sld in self._sliders.items()
            if axis not in self._displayed_axes
        }
        self._controller.incoming_events.emit(
            DimsUpdateEvent(
                source_id=self._id,
                scene_id=self._scene_id,
                slice_indices=updates,
                displayed_axes=None,
            )
        )

    # ── Qt seam 2: push value without re-firing valueChanged ─────────────────

    def _set_value(self, axis: int, value: int) -> None:
        sld = self._sliders[axis]
        sld.blockSignals(True)
        sld.setValue(value)
        sld.blockSignals(False)

    # ── Visibility helper ────────────────────────────────────────────────────

    def _update_visibility(self, displayed_axes: tuple[int, ...]) -> None:
        self._displayed_axes = displayed_axes
        layout: QFormLayout = self._container.layout()
        for axis, sld in self._sliders.items():
            layout.setRowVisible(sld, axis not in displayed_axes)


class QtCanvasWidget:
    """Wraps a render canvas above a ``QtDimsSliders`` panel.

    Composes the two elements in a ``QVBoxLayout`` so that the canvas expands
    to fill available space while the sliders sit below it at a fixed height.

    Prefer constructing via :meth:`from_scene_and_canvas` rather than calling
    ``__init__`` directly.

    Parameters
    ----------
    canvas_view :
        A ``CanvasView`` instance; its ``.widget`` property provides the render
        surface to embed.
    dims_sliders :
        An already-constructed ``QtDimsSliders`` instance.
    parent :
        Optional Qt parent widget.
    """

    def __init__(
        self,
        canvas_view,
        dims_sliders: QtDimsSliders,
        *,
        parent: QWidget | None = None,
    ) -> None:
        self._dims_sliders = dims_sliders

        self._container = QWidget(parent)
        self._container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        layout = QVBoxLayout(self._container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        canvas_qt_widget = canvas_view.widget
        canvas_qt_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(canvas_qt_widget, stretch=1)
        layout.addWidget(dims_sliders.widget)

    @classmethod
    def from_scene_and_canvas(
        cls,
        controller,
        scene,
        canvas_view,
        axis_ranges: dict[int, tuple[int, int]],
        *,
        parent: QWidget | None = None,
    ) -> QtCanvasWidget:
        """Construct from live scene and canvas objects.

        Derives axis labels and the initial dims state from *scene* so that
        callers only need to supply *axis_ranges* (which requires data-store
        knowledge not available on the dims model itself).

        Parameters
        ----------
        controller :
            The ``CellierController`` instance.
        scene :
            The live ``Scene`` object whose dims this panel controls.
        canvas_view :
            The ``CanvasView`` whose ``.widget`` is the render surface.
        axis_ranges :
            Mapping of axis index to ``(min, max)``, e.g.
            ``{0: (0, 99), 1: (0, 511), 2: (0, 511)}``.
        parent :
            Optional Qt parent widget.
        """
        axis_labels = dict(enumerate(scene.dims.coordinate_system.axis_labels))

        selection = scene.dims.selection
        initial_slice_indices = dict(getattr(selection, "slice_indices", {}))
        initial_displayed_axes = getattr(selection, "displayed_axes", ())

        dims_sliders = QtDimsSliders(
            controller=controller,
            scene_id=scene.id,
            axis_ranges=axis_ranges,
            axis_labels=axis_labels,
            initial_slice_indices=initial_slice_indices,
            initial_displayed_axes=initial_displayed_axes,
            parent=parent,
        )
        return cls(canvas_view=canvas_view, dims_sliders=dims_sliders, parent=parent)

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self) -> QWidget:
        """The outer ``QWidget`` to insert into a layout."""
        return self._container

    @property
    def dims_sliders(self) -> QtDimsSliders:
        """The ``QtDimsSliders`` panel embedded below the canvas."""
        return self._dims_sliders

    def close(self) -> None:
        """Unsubscribe the dims sliders from the bus."""
        self._dims_sliders.close()
