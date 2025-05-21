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

import warnings
from abc import abstractmethod
from collections.abc import Container, Hashable, Mapping, Sequence

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QFormLayout, QSizePolicy, QVBoxLayout, QWidget
from superqt import QLabeledSlider
from superqt.utils import signals_blocked

from cellier.models.scene import DimsManager, DimsState
from cellier.types import DimsControlsUpdateEvent, DimsId

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


class QtBaseDimsSliders(QWidget):
    """Base class for dims sliders.

    Dims controls must have the following:
        - a signal named currentIndexChanged that emits a DimsControlsUpdateEvent
          when the dims controls are updated.
        - a method named _on_dims_state_changed that takes a DimsState object
          and updates the dims controls to match the DimsState.
    """

    # signal that gets emitted when the dims controls are changed
    # This is generally registered with the SceneEventBus
    currentIndexChanged = Signal(DimsControlsUpdateEvent)

    @abstractmethod
    def _on_dims_state_changed(self, dims_state: DimsState) -> None:
        """Callback to update the GUI when the dims model's state changes.

        This method is subscribed to the SceneEventBus' dims signal.
        """
        pass


class QtDimsSliders(QtBaseDimsSliders):
    """Widget to control slicing of the dimensions of a scene.

    Parameters
    ----------
    dims_id : DimsId
        The ID of the dims model this widget is associated with.
    parent : QWidget, optional
        The parent widget for this widget.
    """

    def __init__(self, dims_id: DimsId, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._dims_id = dims_id

        self._sliders: dict[Hashable, QLabeledSlider] = {}
        self.setStyleSheet(SLIDER_STYLE)

        layout = QFormLayout(self)
        layout.setSpacing(2)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.setContentsMargins(0, 0, 0, 0)

    @property
    def dims_id(self) -> DimsId:
        """Return the ID of the dims model this widget is associated with."""
        return self._dims_id

    def create_sliders(self, coords: Mapping[Hashable, Sequence]) -> None:
        """Update sliders with the given coordinate ranges."""
        for axis, _coords in coords.items():
            # Create a slider for axis if necessary
            if axis not in self._sliders:
                sld = QLabeledSlider(Qt.Orientation.Horizontal)
                sld.valueChanged.connect(self._on_index_changed)
                self.layout().addRow(str(axis), sld)
                self._sliders[axis] = sld

            # Update axis slider with coordinates
            sld = self._sliders[axis]
            if isinstance(_coords, range):
                sld.setRange(_coords.start, _coords.stop - 1)
                sld.setSingleStep(_coords.step)
            else:
                sld.setRange(0, len(_coords) - 1)
        # self.currentIndexChanged.emit()

    def hide_dimensions(
        self, axes_to_hide: Container[Hashable], show_remainder: bool = True
    ) -> None:
        for ax, slider in self._sliders.items():
            if ax in axes_to_hide:
                self.layout().setRowVisible(slider, False)
            elif show_remainder:
                self.layout().setRowVisible(slider, True)

    def current_index(self) -> Mapping[str, int | slice]:
        """Return the current value of the sliders."""
        return {axis: slider.value() for axis, slider in self._sliders.items()}

    def set_current_index(
        self, value: Mapping[str, int | slice], emit_event: bool = False
    ) -> None:
        """Set the current value of the sliders."""
        changed = False
        # only emit signal if the value actually changed
        # NOTE: this may be unnecessary, since usually the only thing calling
        # set_current_index is the controller, which already knows the value
        # however, we use this method directly in testing and it's nice to ensure.
        with signals_blocked(self):
            for axis, val in value.items():
                if isinstance(val, slice):
                    raise NotImplementedError("Slices are not supported yet")
                if slider := self._sliders.get(axis):
                    if slider.value() != val:
                        changed = True
                        slider.setValue(int(val))
                else:  # pragma: no cover
                    warnings.warn(f"Axis {axis} not found in sliders", stacklevel=2)
        if changed and emit_event:
            self._on_index_changed()

    def _on_dims_state_changed(self, dims_state: DimsState) -> None:
        """Callback to update the GUI when the dims model's state changes.

        This method is subscribed to the SceneEventBus' dims signal.
        """
        dims_point = []
        for axis_index, selection_value in enumerate(
            dims_state.selection.index_selection
        ):
            if isinstance(selection_value, slice):
                low = selection_value.start or dims_state.range[axis_index].start
                high = selection_value.stop or dims_state.range[axis_index].stop
                dims_point.append((low + high) // 2)
            elif isinstance(selection_value, int):
                dims_point.append(selection_value)

        new_index = dict(
            zip(
                dims_state.coordinate_system.axis_labels,
                dims_point,
            )
        )
        self.set_current_index(new_index, emit_event=False)

        # hide the dims that aren't being displayed
        n_displayed_dims = dims_state.selection.n_displayed_dims
        displayed_dimensions = list(dims_state.selection.ordered_dims)[
            -n_displayed_dims:
        ]
        axes_to_hide = [
            dims_state.coordinate_system.axis_labels[i] for i in displayed_dimensions
        ]
        self.hide_dimensions(axes_to_hide)

    def _on_index_changed(self, event: int | None = None) -> None:
        """Handle the index changed event."""
        new_index_selection = []
        for slider in self._sliders.values():
            if slider.isVisible():
                new_index_selection.append(slider.value())
            else:
                new_index_selection.append(slice(None, None, None))

        new_state = {
            "selection": {"index_selection": tuple(new_index_selection)},
        }
        event = DimsControlsUpdateEvent(
            id=self.dims_id,
            state=new_state,
            controls_update_callback=self._on_dims_state_changed,
        )

        self.currentIndexChanged.emit(event)


class QtCanvasWidget(QWidget):
    def __init__(
        self, canvas_widget: QWidget, dims_id: DimsId, parent: QWidget | None = None
    ):
        super().__init__(parent)

        self._canvas_widget = canvas_widget
        self._canvas_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._canvas_widget.setParent(self)

        self._dims_sliders = QtDimsSliders(dims_id=dims_id, parent=self)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout()
        layout.addWidget(self._canvas_widget)
        layout.addWidget(self._dims_sliders)
        self.setLayout(layout)

    @classmethod
    def from_models(cls, render_canvas_widget: QWidget, dims_model: DimsManager):
        # make the widget
        render_canvas_widget.update()
        widget = cls(
            canvas_widget=render_canvas_widget,
            dims_id=dims_model.id,
        )

        # set up the sliders
        dims_ranges = dims_model.range
        sliders_data = {
            axis_label: range(int(start), int(stop), int(step))
            for axis_label, (start, stop, step) in zip(
                dims_model.coordinate_system.axis_labels,
                dims_ranges,
            )
        }
        widget._dims_sliders.create_sliders(sliders_data)

        # set the point to match the current point
        widget._dims_sliders._on_dims_state_changed(dims_state=dims_model.to_state())

        return widget
