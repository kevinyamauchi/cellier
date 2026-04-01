"""AxisInfo — descriptor for one axis of an OME-Zarr array."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class AxisInfo:
    """Descriptor for one axis of an OME-Zarr array.

    Parameters
    ----------
    name : str
        Axis name from OME metadata, e.g. ``"t"``, ``"c"``, ``"z"``.
    unit : str or None
        Physical unit string, or ``None`` if unspecified or not applicable.
    type : str
        OME axis type: ``"space"``, ``"time"``, ``"channel"``, or ``""``.
        Informational only — the DataStore does not gate logic on this field.
    array_dim : int
        Zero-based index of this axis in the full zarr array shape.
        Use to populate ``dims.selection.displayed_axes`` and
        ``dims.selection.slice_indices``.
    """

    name: str
    unit: str | None
    type: str
    array_dim: int
