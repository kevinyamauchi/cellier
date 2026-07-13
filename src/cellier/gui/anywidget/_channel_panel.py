"""anywidget per-channel appearance panel (``ChannelPanel``).

Symmetric to the Qt ``QtChannelList``: a single composite ``WidgetView`` that
drives every channel of one or more multichannel visuals over the
``connect_widget`` bus.  The channel set is fixed at construction (design
section 4), so the panel adds one flattened, synced scalar trait per
``(channel, field)`` via ``add_traits`` (design section 6.3) -- ``ch{i}_{field}``
-- rather than a single list-of-dicts trait.  Each JS control then binds to its
``ch{i}_{field}`` trait exactly like the single-visual panel binds its
appearance traits.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import anywidget
import traitlets
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

_STATIC = Path(__file__).parent / "static"

_DEFAULT_FIELDS: list[str] = ["visible", "color_map", "clim", "opacity"]

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

# field -> traitlets type built from the ChannelAppearance's current value.
_TRAIT_FACTORY = {
    "visible": lambda v: traitlets.Bool(bool(v)),
    "color_map": lambda v: traitlets.Unicode(colormap_to_str(v)),
    "clim": lambda v: traitlets.List([float(v[0]), float(v[1])]),
    "opacity": lambda v: traitlets.Float(float(v)),
    "render_mode_3d": lambda v: traitlets.Unicode(str(v)),
    "iso_threshold": lambda v: traitlets.Float(float(v)),
}


def _parse_trait_name(name: str) -> tuple[int, str]:
    """Parse a ``ch{i}_{field}`` trait name into ``(channel_index, field)``.

    The field part may itself contain underscores (``render_mode_3d``), so the
    split is on the first underscore after the channel index only.
    """
    rest = name[2:]  # strip the "ch" prefix
    idx_str, field = rest.split("_", 1)
    return int(idx_str), field


class ChannelPanel(anywidget.AnyWidget):
    """Bidirectional per-channel appearance controls for the anywidget GUI.

    Parameters
    ----------
    visual_ids :
        UUIDs of the multichannel visuals this panel drives.  A single-panel
        ``Viewer`` passes one id; an ``OrthoViewer`` passes one per panel.  A
        user edit is fanned out to every id; the panel subscribes to all of
        them (design section 6.1).
    channels :
        Mapping of channel index to the initial ``ChannelAppearance`` used to
        seed the per-channel traits.
    clim_range :
        ``(min, max)`` range for the contrast-limits sliders.
    colormap_names :
        Names offered by each colormap select.  Defaults to a curated list.
    fields :
        Which per-channel fields to expose, in display order.  Defaults to
        ``["visible", "color_map", "clim", "opacity"]``.
    channel_labels :
        Optional mapping of channel index to a display label.  Defaults to
        ``"Channel {i}"``.
    """

    _esm = _STATIC / "channel_panel.js"
    _css = _STATIC / "channel_panel.css"

    # psygnal outward signals (the WidgetView contract); not traitlets.
    changed: Signal = Signal(object)
    closed: Signal = Signal()

    # Static config traits (class-level, like the dims panel).
    channel_count = traitlets.Int(0).tag(sync=True)
    channel_labels = traitlets.Dict({}).tag(sync=True)
    fields = traitlets.List([]).tag(sync=True)
    clim_range = traitlets.List([0.0, 1.0]).tag(sync=True)
    colormap_names = traitlets.List([]).tag(sync=True)

    def __init__(
        self,
        visual_ids: list[UUID],
        channels: dict[int, ChannelAppearance],
        *,
        clim_range: tuple[float, float] | list = (0.0, 1.0),
        colormap_names: list[str] | None = None,
        fields: list[str] | None = None,
        channel_labels: dict[int, str] | None = None,
        **kwargs,
    ) -> None:
        resolved_fields = list(fields) if fields is not None else list(_DEFAULT_FIELDS)
        labels_in = channel_labels or {}
        resolved_labels = {str(i): labels_in.get(i, f"Channel {i}") for i in channels}
        super().__init__(
            channel_count=len(channels),
            channel_labels=resolved_labels,
            fields=resolved_fields,
            clim_range=[float(clim_range[0]), float(clim_range[1])],
            colormap_names=colormap_names
            if colormap_names is not None
            else list(_DEFAULT_COLORMAP_NAMES),
            **kwargs,
        )
        self._id = uuid4()
        self._visual_ids = list(visual_ids)
        self._fields = resolved_fields
        self._applying = False

        # Flattened per-(channel, field) traits (design section 6.3).
        per_channel = {
            f"ch{i}_{field}": _TRAIT_FACTORY[field](getattr(ch, field))
            for i, ch in channels.items()
            for field in resolved_fields
        }
        self.add_traits(**{k: t.tag(sync=True) for k, t in per_channel.items()})
        self._per_channel_names = list(per_channel)
        self.observe(self._on_trait_change, names=self._per_channel_names)

    # ------------------------------------------------------------------
    # WidgetView contract
    # ------------------------------------------------------------------

    @property
    def widget(self) -> ChannelPanel:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

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

    def _on_trait_change(self, change) -> None:
        if self._applying:
            return  # bus -> widget write; do not echo back
        channel_index, field = _parse_trait_name(change.name)
        value = tuple(change.new) if field == "clim" else change.new
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

    # ------------------------------------------------------------------
    # model -> widget (inbound)
    # ------------------------------------------------------------------

    def _on_changed(self, event: ChannelAppearanceChangedEvent) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        field = event.field_name
        trait_name = f"ch{event.channel_index}_{field}"
        if not self.has_trait(trait_name):
            return  # a field this panel does not display
        value = event.new_value
        if field == "color_map":
            value = colormap_to_str(value)
        elif field == "clim":
            value = [float(value[0]), float(value[1])]
        self._applying = True
        try:
            setattr(self, trait_name, value)
        finally:
            self._applying = False
