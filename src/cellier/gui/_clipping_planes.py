"""The "Clipping planes" control of a visual, decided once for both toolkits.

A visual carries a tuple of :class:`~cellier.visuals.ClippingPlane` in its
level-0 data coordinates.  The Qt and anywidget controls
(``QtClippingPlanesControls``, ``AnywidgetClippingPlanesControls``) draw one
row per plane: an enabled checkbox, a normal (a preset per data axis, or
typed), a flip button, a position slider along the normal and a remove
button; an add button appends a plane.  This module holds the rows, turns
them into planes and back, and carries edits to the bus and model changes
back.

A row is plain data, so the anywidget control syncs the whole list in one
trait and constructs nothing when a plane is added::

    {"enabled": True, "normal": [0.0, 0.0, 1.0], "position": 12.0}

``normal`` has one entry per data axis.  ``position`` is where the plane
sits along its unit normal: the plane is ``normal . p == position *
|normal|``.  Values are in data units (voxels for an image).

Nothing here imports a toolkit.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any
from uuid import UUID

from cellier.events import (
    ClippingPlanesChangedEvent,
    ClippingPlanesUpdateEvent,
    SubscriptionSpec,
)
from cellier.gui._loading import error_message

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

CLIPPING_PLANES_TITLE = "Clipping planes"
"""The control's name: its ``DEFAULT_TITLE`` on both toolkits."""

CUSTOM_PRESET = "custom"
"""The normal preset shown when the normal is not along one data axis."""

Row = dict[str, Any]


def _length(normal: Sequence[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in normal))


def rows_from_planes(planes: Iterable[Any]) -> list[Row]:
    """The rows that show *planes* (a visual's ``clipping_planes``)."""
    rows = []
    for item in planes:
        normal = [float(v) for v in item.plane.normal]
        rows.append(
            {
                "enabled": bool(item.enabled),
                "normal": normal,
                "position": float(item.plane.offset) / _length(normal),
            }
        )
    return rows


def planes_from_rows(rows: Iterable[Mapping[str, Any]], coordinate_system: UUID):
    """Build the ``ClippingPlane`` tuple that *rows* describe.

    Raises
    ------
    ValueError
        If a row's normal is zero or not finite.
    """
    from cellier.transform import Plane
    from cellier.visuals import ClippingPlane

    planes = []
    for row in rows:
        normal = [float(v) for v in row["normal"]]
        planes.append(
            ClippingPlane(
                plane=Plane(
                    coordinate_system=coordinate_system,
                    normal=normal,
                    offset=float(row["position"]) * _length(normal),
                ),
                enabled=bool(row.get("enabled", True)),
            )
        )
    return tuple(planes)


def normalized_rows(rows: Iterable[Mapping[str, Any]]) -> list[Row]:
    """Rows as plain Python values, so two lists compare by value."""
    return [
        {
            "enabled": bool(row.get("enabled", True)),
            "normal": [float(v) for v in row["normal"]],
            "position": float(row["position"]),
        }
        for row in rows
    ]


def position_range(
    normal: Sequence[float], bounds: Sequence[Sequence[float]]
) -> tuple[float, float]:
    """The slider's range: the data bounding box projected onto the normal.

    Parameters
    ----------
    normal : Sequence[float]
        One entry per data axis.
    bounds : Sequence[Sequence[float]]
        ``(low, high)`` per data axis.  An axis the normal has no component
        on does not matter.

    Returns
    -------
    tuple[float, float]
        Positions along the unit normal at which the plane touches the
        box; ``(0.0, 1.0)`` for a zero normal.
    """
    length = _length(normal)
    if length == 0.0:
        return 0.0, 1.0
    low = high = 0.0
    for component, (axis_low, axis_high) in zip(normal, bounds):
        unit = float(component) / length
        if unit == 0.0:
            continue
        a, b = unit * float(axis_low), unit * float(axis_high)
        low += min(a, b)
        high += max(a, b)
    if high <= low:
        high = low + 1.0
    return low, high


def preset_of(normal: Sequence[float], axis_names: Sequence[str]) -> str:
    """The axis name a normal lies along, or :data:`CUSTOM_PRESET`."""
    along = [index for index, value in enumerate(normal) if float(value) != 0.0]
    if len(along) == 1:
        return axis_names[along[0]]
    return CUSTOM_PRESET


def preset_normal(
    preset: str, axis_names: Sequence[str], current: Sequence[float]
) -> list[float]:
    """The unit normal along axis *preset*, keeping *current*'s direction."""
    index = list(axis_names).index(preset)
    sign = -1.0 if float(current[index]) < 0 else 1.0
    return [sign if i == index else 0.0 for i in range(len(axis_names))]


def parse_normal(text: str, ndim: int) -> list[float]:
    """Read a typed normal: *ndim* numbers separated by commas or spaces.

    Raises
    ------
    ValueError
        If the count is wrong, an entry is not a number, or all are zero.
    """
    parts = [part for part in text.replace(",", " ").split() if part]
    if len(parts) != ndim:
        raise ValueError(f"A normal here has {ndim} entries; got {len(parts)}.")
    normal = [float(part) for part in parts]
    if not any(value != 0.0 for value in normal) or not all(
        math.isfinite(value) for value in normal
    ):
        raise ValueError("A normal must be finite and not all zero.")
    return normal


def new_row(axis_names: Sequence[str], bounds: Sequence[Sequence[float]]) -> Row:
    """A plane across the last data axis, through the middle of the data."""
    normal = [0.0] * len(axis_names)
    normal[-1] = 1.0
    low, high = position_range(normal, bounds)
    return {"enabled": True, "normal": normal, "position": 0.5 * (low + high)}


def clipping_planes_seed(visual: Any, store: Any) -> dict[str, Any]:
    """What a clipping planes control is built with, read off the models.

    Parameters
    ----------
    visual : BaseVisual
        The visual (for an ``OrthoViewer`` group, any one of them: they
        read one store and carry the same planes).
    store : BaseDataStore
        The store the visual reads.

    Returns
    -------
    dict[str, Any]
        ``coordinate_system`` (the store's level-0 system id, as a string),
        ``axis_names``, ``bounds`` (``[low, high]`` per data axis) and
        ``planes`` (the rows).
    """
    system = store.data_coordinate_system
    extents = store.axis_extents
    names = [str(name) for name in system.axis_names()]
    if extents is None:
        bounds = [[0.0, 1.0] for _ in names]
    else:
        bounds = [[float(low), float(high)] for low, high in extents]
    return {
        "coordinate_system": str(system.id),
        "axis_names": names,
        "bounds": bounds,
        "planes": rows_from_planes(visual.clipping_planes),
    }


class ClippingPlanesEditor:
    """The toolkit-neutral half of a clipping planes control.

    Holds the rows the model last reported, sends edits, and tells the
    widget what to show.  An edit is one ``ClippingPlanesUpdateEvent`` per
    visual, carrying the whole new tuple.  If the controller refuses it,
    the widget is told to show the last rows again, with the error.

    Parameters
    ----------
    visual_ids : Iterable[UUID]
        The visuals edited together (an ``OrthoViewer`` panel group).
    coordinate_system : UUID or str
        The level-0 data coordinate system of the store they read.
    axis_names : Sequence[str]
        Its axis names, in order.
    bounds : Sequence[Sequence[float]]
        ``(low, high)`` of the data on each axis, for the position range.
    planes : Sequence[Mapping[str, Any]]
        The current rows.
    source_id : UUID
        The widget's id, stamped on each edit.
    emit : Callable[[ClippingPlanesUpdateEvent], None]
        Sends an edit (the widget's ``changed.emit``).
    show : Callable[[list[Row], str], None]
        Draws the rows and an error message ("" for none).
    """

    def __init__(
        self,
        visual_ids: Iterable[UUID],
        coordinate_system: UUID | str,
        axis_names: Sequence[str],
        bounds: Sequence[Sequence[float]],
        planes: Sequence[Mapping[str, Any]],
        source_id: UUID,
        emit: Callable[[ClippingPlanesUpdateEvent], None],
        show: Callable[[list[Row], str], None],
    ) -> None:
        self._visual_ids = tuple(visual_ids)
        self._coordinate_system = UUID(str(coordinate_system))
        self.axis_names = [str(name) for name in axis_names]
        self.bounds = [[float(low), float(high)] for low, high in bounds]
        self.rows: list[Row] = normalized_rows(planes)
        self._source_id = source_id
        self._emit = emit
        self._show = show

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One ``ClippingPlanesChangedEvent`` subscription per visual."""
        return [
            SubscriptionSpec(
                ClippingPlanesChangedEvent, self.on_changed, entity_id=visual_id
            )
            for visual_id in self._visual_ids
        ]

    # -- edits ---------------------------------------------------------------

    def set_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Send *rows* as the new planes; nothing when they are the current.

        The comparison is by value: a host that delivers one front-end edit
        twice (marimo does) sends one update.
        """
        try:
            rows = normalized_rows(rows)
            if rows == self.rows:
                return
            planes = planes_from_rows(rows, self._coordinate_system)
            for visual_id in self._visual_ids:
                self._emit(
                    ClippingPlanesUpdateEvent(
                        source_id=self._source_id,
                        visual_id=visual_id,
                        clipping_planes=planes,
                    )
                )
        except Exception as error:
            self._show(list(self.rows), error_message(error))
            return
        self.rows = rows
        self._show(list(self.rows), "")

    def _edited(self, index: int, **changes: Any) -> list[Row]:
        rows = [dict(row) for row in self.rows]
        rows[index].update(changes)
        return rows

    def add(self) -> None:
        """Append a plane across the last axis, through the data's middle."""
        self.set_rows([*self.rows, new_row(self.axis_names, self.bounds)])

    def remove(self, index: int) -> None:
        """Remove plane *index*."""
        self.set_rows([row for i, row in enumerate(self.rows) if i != index])

    def set_enabled(self, index: int, enabled: bool) -> None:
        """Switch plane *index* on or off; it stays in the list."""
        self.set_rows(self._edited(index, enabled=bool(enabled)))

    def set_position(self, index: int, position: float) -> None:
        """Move plane *index* along its normal."""
        self.set_rows(self._edited(index, position=float(position)))

    def flip(self, index: int) -> None:
        """Keep the other side of plane *index*; the plane does not move."""
        row = self.rows[index]
        self.set_rows(
            self._edited(
                index,
                normal=[-v if v != 0.0 else 0.0 for v in row["normal"]],
                position=-row["position"],
            )
        )

    def set_normal(self, index: int, normal: Sequence[float]) -> None:
        """Give plane *index* a new normal, through the point it passed.

        The plane keeps the point of the old plane nearest the middle of
        the data, so changing its direction turns it in place.
        """
        row = self.rows[index]
        old = row["normal"]
        old_length = _length(old)
        centre = [0.5 * (low + high) for low, high in self.bounds]
        # The old plane's point nearest the centre.
        distance = (
            sum(n * c for n, c in zip(old, centre)) / old_length - row["position"]
        )
        anchor = [c - distance * n / old_length for n, c in zip(old, centre)]
        normal = [float(v) for v in normal]
        length = _length(normal)
        if length == 0.0:
            self._show(list(self.rows), "A normal must not be all zero.")
            return
        position = sum(n * a for n, a in zip(normal, anchor)) / length
        self.set_rows(self._edited(index, normal=normal, position=position))

    def set_preset(self, index: int, preset: str) -> None:
        """Point plane *index* along data axis *preset*."""
        if preset == CUSTOM_PRESET:
            return
        normal = preset_normal(preset, self.axis_names, self.rows[index]["normal"])
        self.set_normal(index, normal)

    def set_normal_text(self, index: int, text: str) -> None:
        """Give plane *index* a typed normal; a bad entry shows an error."""
        try:
            normal = parse_normal(text, len(self.axis_names))
        except ValueError as error:
            self._show(list(self.rows), str(error))
            return
        self.set_normal(index, normal)

    # -- model -> widget -----------------------------------------------------

    def on_changed(self, event: ClippingPlanesChangedEvent) -> None:
        """The model changed (this widget's edit or anyone's): show it."""
        rows = rows_from_planes(event.clipping_planes)
        if rows == self.rows:
            return
        self.rows = rows
        self._show(list(self.rows), "")

    def describe(self) -> list[Row]:
        """The rows with what a front end needs to draw each one.

        Adds ``preset``, ``normal_text`` and the slider's ``low`` / ``high``.
        """
        described = []
        for row in self.rows:
            low, high = position_range(row["normal"], self.bounds)
            described.append(
                {
                    **row,
                    "preset": preset_of(row["normal"], self.axis_names),
                    "normal_text": ", ".join(f"{v:g}" for v in row["normal"]),
                    "low": min(low, row["position"]),
                    "high": max(high, row["position"]),
                }
            )
        return described
