"""The Qt "Clipping planes" control of a visual.

One row per plane, described and edited by
:class:`cellier.gui._clipping_planes.ClippingPlanesEditor`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from psygnal import Signal

from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui._clipping_planes import (
    CLIPPING_PLANES_TITLE,
    CUSTOM_PRESET,
    ClippingPlanesEditor,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

    from cellier.events import SubscriptionSpec

#: Steps of the position slider (Qt sliders are integers).
_SLIDER_STEPS = 1000


class _PlaneRow:
    """The widgets of one plane.  Updated in place; never rebuilt by a move."""

    def __init__(self, owner: QtClippingPlanesControls, index: int, parent) -> None:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QGridLayout,
            QLineEdit,
            QPushButton,
            QSlider,
            QWidget,
        )

        self.index = index
        self._range = (0.0, 1.0)
        editor = owner._editor
        self.widget = QWidget(parent)
        grid = QGridLayout(self.widget)
        grid.setContentsMargins(0, 2, 0, 2)

        self.enabled = QCheckBox(self.widget)
        self.enabled.setToolTip("Whether this plane clips. It stays in the list.")
        self.enabled.toggled.connect(lambda v: editor.set_enabled(self.index, v))

        self.preset = QComboBox(self.widget)
        self.preset.addItems([*editor.axis_names, CUSTOM_PRESET])
        self.preset.setToolTip("The data axis the plane's normal points along.")
        self.preset.currentTextChanged.connect(
            lambda v: editor.set_preset(self.index, v)
        )

        self.flip = QPushButton("Flip", self.widget)
        self.flip.setToolTip("Keep the other side of the plane.")
        self.flip.clicked.connect(lambda: editor.flip(self.index))

        self.remove = QPushButton("Remove", self.widget)
        self.remove.clicked.connect(lambda: editor.remove(self.index))

        self.normal = QLineEdit(self.widget)
        self.normal.setToolTip(
            "The normal, one number per data axis ("
            + ", ".join(editor.axis_names)
            + "). It points to the side that is kept. Data units."
        )
        self.normal.editingFinished.connect(
            lambda: editor.set_normal_text(self.index, self.normal.text())
        )

        self.slider = QSlider(Qt.Orientation.Horizontal, self.widget)
        self.slider.setRange(0, _SLIDER_STEPS)
        self.slider.setToolTip("Where the plane sits along its normal. Data units.")
        self.slider.valueChanged.connect(self._on_slider)

        self.position = QDoubleSpinBox(self.widget)
        self.position.setDecimals(2)
        self.position.setRange(-1e9, 1e9)
        self.position.setKeyboardTracking(False)
        self.position.valueChanged.connect(lambda v: editor.set_position(self.index, v))

        grid.addWidget(self.enabled, 0, 0)
        grid.addWidget(self.preset, 0, 1)
        grid.addWidget(self.normal, 0, 2)
        grid.addWidget(self.flip, 0, 3)
        grid.addWidget(self.remove, 0, 4)
        grid.addWidget(self.slider, 1, 1, 1, 2)
        grid.addWidget(self.position, 1, 3, 1, 2)
        grid.setColumnStretch(2, 1)
        self._editor = editor
        self._inputs = (
            self.enabled,
            self.preset,
            self.normal,
            self.slider,
            self.position,
        )

    def _on_slider(self, step: int) -> None:
        low, high = self._range
        self._editor.set_position(self.index, low + (high - low) * step / _SLIDER_STEPS)

    def show(self, row: Mapping[str, Any]) -> None:
        """Show *row* (from ``ClippingPlanesEditor.describe``); emits nothing."""
        for widget in self._inputs:
            widget.blockSignals(True)
        try:
            self.enabled.setChecked(bool(row["enabled"]))
            self.preset.setCurrentText(row["preset"])
            if not self.normal.hasFocus():
                self.normal.setText(row["normal_text"])
            low, high = float(row["low"]), float(row["high"])
            self._range = (low, high)
            span = high - low
            step = (
                0
                if span <= 0
                else round((row["position"] - low) / span * _SLIDER_STEPS)
            )
            self.slider.setValue(int(min(max(step, 0), _SLIDER_STEPS)))
            self.position.setSingleStep(max(span / 200.0, 0.01))
            self.position.setValue(float(row["position"]))
        finally:
            for widget in self._inputs:
                widget.blockSignals(False)


class QtClippingPlanesControls(VisualIdGroup):
    """A visual's clipping planes: one row per plane, and an add button.

    Each row has an enabled checkbox, the data axis the normal points along
    (or ``custom``), the normal itself, a flip button, a position slider
    along the normal and a remove button.  Values are in the visual's data
    coordinates.  An edit is sent as ``ClippingPlanesUpdateEvent`` with the
    whole new tuple.  A moved plane updates its row in place, so a slider is
    not destroyed while it is dragged.

    Wire to the controller after construction::

        seed = clipping_planes_seed(visual, store)
        controls = QtClippingPlanesControls(visual.id, **seed)
        controller.connect_widget(
            controls, subscription_specs=controls.subscription_specs()
        )

    Parameters
    ----------
    visual_id :
        The visual, or a group of them (an ``OrthoViewer``'s panel
        siblings), edited together.
    coordinate_system :
        The level-0 data coordinate system of the store they read (its id).
    axis_names :
        The data axis names, in order.
    bounds :
        ``(low, high)`` of the data on each axis.
    planes :
        The current rows (``rows_from_planes(visual.clipping_planes)``).
    title :
        The group's name.  Defaults to :data:`DEFAULT_TITLE`.
    parent :
        Optional Qt parent widget.
    """

    DEFAULT_TITLE = CLIPPING_PLANES_TITLE
    """Name shown when no ``title=`` is given."""

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        coordinate_system: UUID | str,
        axis_names: Sequence[str],
        bounds: Sequence[Sequence[float]],
        planes: Sequence[Mapping[str, Any]] = (),
        title: str | None = None,
        parent=None,
    ) -> None:
        from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

        from cellier.gui.qt.visuals._chrome import titled_group

        self._id = uuid4()
        self._init_visual_ids(visual_id)

        self._content = QWidget(parent)
        column = QVBoxLayout(self._content)
        column.setContentsMargins(0, 0, 0, 0)
        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        column.addLayout(self._rows_layout)
        self._rows: list[_PlaneRow] = []

        self._add = QPushButton("Add plane", self._content)
        self._add.setToolTip("Add a plane across the last axis, through the middle.")
        column.addWidget(self._add)

        self._error = QLabel(self._content)
        self._error.setWordWrap(True)
        self._error.setStyleSheet("color: #d04040")
        self._error.setVisible(False)
        column.addWidget(self._error)

        self._group = titled_group(
            self.DEFAULT_TITLE if title is None else title, self._content, parent
        )
        self._editor = ClippingPlanesEditor(
            self._visual_ids,
            coordinate_system,
            axis_names,
            bounds,
            planes,
            self._id,
            self.changed.emit,
            self._show,
        )
        self._add.clicked.connect(lambda: self._editor.add())
        self._show(self._editor.rows, "")

    # -- Public interface ------------------------------------------------------

    @property
    def widget(self):
        """The widget to insert into a layout: the titled group."""
        return self._group

    @property
    def planes(self) -> list[dict[str, Any]]:
        """The rows as last reported by the model."""
        return [dict(row) for row in self._editor.rows]

    @property
    def editor(self) -> ClippingPlanesEditor:
        """The toolkit-neutral half (for tests and scripting)."""
        return self._editor

    @property
    def error(self) -> str:
        """The reason the last edit was refused, or ""."""
        return self._error.text()

    def row(self, index: int) -> _PlaneRow:
        """The widgets of plane *index* (for tests and scripting)."""
        return self._rows[index]

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One ``ClippingPlanesChangedEvent`` per visual."""
        return self._editor.subscription_specs()

    # -- model -> widget -------------------------------------------------------

    def _show(self, rows: list[dict[str, Any]], error: str) -> None:
        described = self._editor.describe()
        while len(self._rows) > len(described):
            row = self._rows.pop()
            self._rows_layout.removeWidget(row.widget)
            row.widget.setParent(None)
            row.widget.deleteLater()
        while len(self._rows) < len(described):
            row = _PlaneRow(self, len(self._rows), self._content)
            self._rows.append(row)
            self._rows_layout.addWidget(row.widget)
        for row, values in zip(self._rows, described):
            row.show(values)
        self._error.setText(error)
        self._error.setVisible(bool(error))
