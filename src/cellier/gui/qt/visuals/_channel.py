"""Per-channel appearance controls (``QtChannelList``) wired to the event bus.

A single composite ``WidgetView`` that drives every channel of one or more
multichannel visuals.  It is a port of oz-viewer's ``build_channel_group`` /
``build_channel_list_widget`` converted from direct model binding to the
``connect_widget`` bus contract: one ``changed`` / ``closed`` pair, an
aggregated ``subscription_specs()`` (one per visual id), and echo filtering on
``source_id``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from psygnal import Signal

from cellier.events import (
    ChannelAppearanceChangedEvent,
    ChannelAppearanceUpdateEvent,
    SubscriptionSpec,
)
from cellier.gui._colormap_util import colormap_to_str

if TYPE_CHECKING:
    from uuid import UUID

    from cellier.visuals._channel_appearance import ChannelAppearance

_DEFAULT_FIELDS: tuple[str, ...] = ("visible", "color_map", "clim", "opacity")

_DEFAULT_COLORMAP_NAMES: list[str] = [
    "viridis",
    "plasma",
    "grays",
    "magma",
    "inferno",
    "cividis",
    "red",
    "green",
    "blue",
    "magenta",
    "cyan",
]


class QtChannelList:
    """Bidirectional per-channel appearance controls on the cellier bus.

    Parameters
    ----------
    visual_ids :
        UUIDs of the multichannel visuals this widget drives.  A single-panel
        ``Viewer`` passes one id; an ``OrthoViewer`` passes one per panel.  A
        user edit is fanned out to every id; the widget subscribes to all of
        them (design section 6.1).
    channels :
        Mapping of channel index to the initial ``ChannelAppearance`` used to
        populate the controls.
    clim_range :
        ``(min, max)`` range for the contrast-limits sliders.
    colormap_names :
        Names offered by each colormap combo.  Defaults to a curated list.
    fields :
        Which per-channel fields to expose, in display order.  Defaults to
        ``("visible", "color_map", "clim", "opacity")``.
    channel_labels :
        Optional mapping of channel index to a display label.  Defaults to
        ``"Channel {i}"``.
    decimals :
        Decimal places shown on the clim / opacity sliders.
    parent :
        Optional Qt parent widget.
    """

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_ids: list[UUID],
        channels: dict[int, ChannelAppearance],
        *,
        clim_range: tuple[float, float] = (0.0, 1.0),
        colormap_names: list[str] | None = None,
        fields: tuple[str, ...] | None = None,
        channel_labels: dict[int, str] | None = None,
        decimals: int = 2,
        parent=None,
    ) -> None:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (
            QCheckBox,
            QGroupBox,
            QScrollArea,
            QVBoxLayout,
            QWidget,
        )
        from superqt import (
            QColormapComboBox,
            QLabeledDoubleRangeSlider,
            QLabeledDoubleSlider,
        )

        self._id = uuid4()
        self._visual_ids = list(visual_ids)
        self._fields = tuple(fields) if fields is not None else _DEFAULT_FIELDS
        self._clim_range = clim_range
        self._colormap_names = (
            list(colormap_names)
            if colormap_names is not None
            else list(_DEFAULT_COLORMAP_NAMES)
        )
        self._channel_labels = channel_labels or {}
        self._decimals = decimals

        # (channel_index, field) -> callable that applies a value with the
        # control's signals blocked (used by the inbound bus handler).
        self._appliers: dict[tuple[int, str], object] = {}

        container = QWidget(parent)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        for channel_index, appearance in channels.items():
            label = self._channel_labels.get(channel_index, f"Channel {channel_index}")
            group = QGroupBox(label)
            group_layout = QVBoxLayout(group)

            for field in self._fields:
                if field == "visible":
                    control = QCheckBox("Visible")
                    control.setChecked(bool(appearance.visible))
                    control.toggled.connect(self._make_emit(channel_index, "visible"))
                    self._appliers[(channel_index, "visible")] = (
                        self._make_checkbox_applier(control)
                    )
                    group_layout.addWidget(control)
                elif field == "color_map":
                    control = QColormapComboBox()
                    control.addColormaps(self._colormap_names)
                    control.setCurrentColormap(appearance.color_map)
                    control.currentColormapChanged.connect(
                        self._make_emit(channel_index, "color_map")
                    )
                    self._appliers[(channel_index, "color_map")] = (
                        self._make_colormap_applier(control)
                    )
                    group_layout.addWidget(control)
                elif field == "clim":
                    control = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
                    control.setDecimals(self._decimals)
                    control.setRange(*self._clim_range)
                    control.setValue(tuple(appearance.clim))
                    control.valueChanged.connect(self._make_clim_emit(channel_index))
                    self._appliers[(channel_index, "clim")] = self._make_slider_applier(
                        control
                    )
                    group_layout.addWidget(control)
                elif field == "opacity":
                    control = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
                    control.setRange(0.0, 1.0)
                    control.setSingleStep(0.05)
                    control.setValue(float(appearance.opacity))
                    control.valueChanged.connect(
                        self._make_emit(channel_index, "opacity")
                    )
                    self._appliers[(channel_index, "opacity")] = (
                        self._make_slider_applier(control)
                    )
                    group_layout.addWidget(control)

            container_layout.addWidget(group)

        if len(channels) > 3:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setWidget(container)
            self._widget = scroll
        else:
            self._widget = container

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def widget(self):
        """The Qt widget (container or scroll area) to embed in a layout."""
        return self._widget

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """Return one inbound subscription per visual id (subscribe-to-all)."""
        return [
            SubscriptionSpec(
                event_type=ChannelAppearanceChangedEvent,
                handler=self._on_changed,
                entity_id=vid,
            )
            for vid in self._visual_ids
        ]

    # ------------------------------------------------------------------
    # widget -> model (outbound)
    # ------------------------------------------------------------------

    def _emit(self, channel_index: int, field: str, value) -> None:
        for vid in self._visual_ids:
            self.changed.emit(
                ChannelAppearanceUpdateEvent(
                    source_id=self._id,
                    visual_id=vid,
                    channel_index=channel_index,
                    field=field,
                    value=value,
                )
            )

    def _make_emit(self, channel_index: int, field: str):
        def _on_change(value) -> None:
            self._emit(channel_index, field, value)

        return _on_change

    def _make_clim_emit(self, channel_index: int):
        def _on_change(value) -> None:
            self._emit(channel_index, "clim", tuple(value))

        return _on_change

    # ------------------------------------------------------------------
    # model -> widget (inbound)
    # ------------------------------------------------------------------

    def _on_changed(self, event: ChannelAppearanceChangedEvent) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        applier = self._appliers.get((event.channel_index, event.field_name))
        if applier is None:
            return  # a field this widget does not display
        value = event.new_value
        if event.field_name == "color_map":
            value = colormap_to_str(value)
        elif event.field_name == "clim":
            value = [float(value[0]), float(value[1])]
        applier(value)

    # ------------------------------------------------------------------
    # Appliers: set a control's value without re-firing its change signal.
    # ------------------------------------------------------------------

    @staticmethod
    def _make_checkbox_applier(control):
        def _apply(value) -> None:
            control.blockSignals(True)
            control.setChecked(bool(value))
            control.blockSignals(False)

        return _apply

    @staticmethod
    def _make_colormap_applier(control):
        def _apply(value) -> None:
            control.blockSignals(True)
            control.setCurrentColormap(value)
            control.blockSignals(False)

        return _apply

    @staticmethod
    def _make_slider_applier(control):
        def _apply(value) -> None:
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)

        return _apply
