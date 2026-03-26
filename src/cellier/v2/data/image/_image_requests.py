"""ChunkRequest — immutable request object for a single padded brick read.

Lives here as a standalone module so both ``MultiscaleZarrDataStore`` and
``AsyncSlicer`` can import it without a circular dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from uuid import UUID


class ChunkRequest(NamedTuple):
    """A request for one padded brick / chunk of data.

    All coordinates encode the *padded* region including overlap.  They
    may be negative or exceed the store bounds; ``get_data()`` on the
    ``MultiscaleZarrDataStore`` is responsible for clamping and
    zero-padding so the returned array always has the full requested
    shape ``(z_stop-z_start, y_stop-y_start, x_stop-x_start)`` for 3D
    requests, or ``(y_stop-y_start, x_stop-x_start)`` for 2D requests
    (when ``z_slice`` is set).

    Parameters
    ----------
    chunk_request_id :
        Unique ID for this individual chunk.  Used by the callback to look
        up the corresponding ``TileSlot`` in the ``slot_map`` closure.
    slice_request_id :
        Shared ID for all chunks that belong to the same planning event
        (one Update press → one ``slice_request_id``).  Used by
        ``AsyncSlicer.cancel()`` to cancel all in-flight work for a
        single update cycle.
    scale_index :
        0-based index into ``MultiscaleZarrDataStore`` levels
        (0 = finest).  Note: this is ``scale_index``, not
        ``BrickKey.level`` (which is 1-based).
    z_start, y_start, x_start :
        Padded region origin in voxels at this scale level.  May be
        negative for boundary bricks.
    z_stop, y_stop, x_stop :
        Padded region end (exclusive) in voxels.  May exceed store
        bounds for boundary bricks.
    z_slice :
        If not ``None``, this is a 2D request: the data store reads a
        single z-plane at this index (at the current scale level) and
        returns a 2D ``(H, W)`` array.  ``z_start`` and ``z_stop`` are
        ignored when ``z_slice`` is set.
    """

    chunk_request_id: UUID
    slice_request_id: UUID
    scale_index: int
    z_start: int
    y_start: int
    x_start: int
    z_stop: int
    y_stop: int
    x_stop: int
    z_slice: int | None = None
