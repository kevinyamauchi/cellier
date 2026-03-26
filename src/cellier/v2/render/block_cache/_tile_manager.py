"""Tile manager with LRU eviction and late tilemap insertion.

Key design principle: ``tilemap`` is a *committed* state only.
``stage()`` allocates slots but does **not** insert them into ``tilemap``
— it puts them in ``_in_flight`` instead.  Only ``commit()`` moves a
brick from ``_in_flight`` into ``tilemap``, and only after the data has
been written to the GPU cache.  This means:

* ``rebuild_lut()`` only ever sees bricks with valid data.
* Cancelling an async load is clean: call ``release_all_in_flight()``
  and the reserved-but-never-written slots are returned to the free list
  without touching ``tilemap``.

LRU eviction with a min-heap
-----------------------------
A min-heap keeps the entry with the smallest timestamp at index 0, so
finding the LRU victim is O(1) and removing it is O(log N).  This reduces
``stage()`` from O(misses x N) to O(misses x log N).

Lazy deletion
-------------
Cache hits update a slot's timestamp in ``tilemap`` but do not update the
heap, because finding and fixing the heap entry would itself be O(N).
Instead the heap holds **stale entries** whose recorded timestamp no
longer matches the slot's current timestamp.  ``_evict_lru`` discards
stale entries lazily when it pops them.  Each entry is pushed once and
popped once, so all operations are O(log N) amortised.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cellier.v2.logging import _CACHE_LOGGER

if TYPE_CHECKING:
    from cellier.v2.render.block_cache._cache_parameters_3d import (
        BlockCacheParameters3D,
    )


@dataclass(frozen=True)
class BlockKey3D:
    """Identifier for a brick at a specific LOAD level.

    Attributes
    ----------
    level : int
        1-indexed LOAD level (1 = finest).
    gz, gy, gx : int
        Grid position at this level's resolution.
    """

    level: int
    gz: int
    gy: int
    gx: int


@dataclass
class TileSlot:
    """A slot in the GPU cache.

    Attributes
    ----------
    index : int
        Flat slot index (0 = reserved, never used for data).
    grid_pos : tuple[int, int, int]
        ``(sz, sy, sx)`` in the cache grid.
    timestamp : int
        Frame number when last accessed (for LRU).
    """

    index: int
    grid_pos: tuple[int, int, int]
    timestamp: int = 0


class TileManager:
    """Manages brick-to-slot mapping with LRU eviction and late insertion.

    Slot lifecycle
    --------------
    1. ``stage()``                 -- allocates a slot; records it in
                                      ``_in_flight`` (NOT in ``tilemap``).
    2. ``commit()``                -- called after data is written to the
                                      GPU cache; moves the entry from
                                      ``_in_flight`` into ``tilemap`` and
                                      pushes it onto the LRU heap.
    3. ``release_all_in_flight()`` -- called on cancellation; returns all
                                      reserved-but-not-yet-committed slots
                                      to ``free_slots``.  ``tilemap`` is
                                      untouched, so only valid bricks
                                      remain renderable.

    Parameters
    ----------
    cache_parameters :
        Cache sizing metadata (grid dimensions, slot count, etc.).
    """

    def __init__(self, cache_parameters: BlockCacheParameters3D) -> None:
        self.cache_parameters = cache_parameters

        # brick -> slot  (committed, renderable bricks only)
        self.tilemap: dict[BlockKey3D, TileSlot] = {}

        # slot index -> brick  (committed bricks only; None = free or in-flight)
        self.slot_index: dict[int, BlockKey3D | None] = {
            i: None for i in range(cache_parameters.n_slots)
        }
        # Slot 0 is reserved (samples as black / out-of-bounds).
        self.slot_index[0] = BlockKey3D(level=0, gz=0, gy=0, gx=0)

        # Free slots (everything except slot 0).
        self.free_slots: list[int] = list(range(cache_parameters.n_slots - 1, 0, -1))

        # slot index -> brick  (allocated by stage() but not yet committed)
        # These slots are occupied (not in free_slots) but not renderable.
        self._in_flight: dict[int, BlockKey3D] = {}

        # Min-heap of (timestamp, slot_index) for LRU eviction.
        # Only committed slots are pushed onto the heap.
        self._lru_heap: list[tuple[int, int]] = []

    # -- Slot lifecycle --------------------------------------------------

    def stage(
        self,
        required_bricks: dict[BlockKey3D, int],
        frame_number: int,
    ) -> list[tuple[BlockKey3D, TileSlot]]:
        """Process required bricks: mark hits, reserve slots for misses.

        Hits update the timestamp of the existing committed slot.
        Misses allocate a slot (evicting LRU if necessary) and record it
        in ``_in_flight``.  The slot is **not** added to ``tilemap`` here
        -- only ``commit()`` does that.

        Parameters
        ----------
        required_bricks :
            Mapping from brick key to desired level (the level is also
            encoded in the key itself).
        frame_number :
            Current frame number for LRU timestamps.

        Returns
        -------
        fill_plan :
            ``(BlockKey3D, TileSlot)`` pairs for bricks that need loading,
            in the same order as the misses in ``required_bricks``.
        """
        miss_list: list[BlockKey3D] = []

        for brick_key in required_bricks:
            if brick_key in self.tilemap:
                # Hit -- refresh timestamp; push a new heap entry (the old
                # one becomes stale and is discarded lazily in _evict_lru).
                slot = self.tilemap[brick_key]
                slot.timestamp = frame_number
                heapq.heappush(self._lru_heap, (frame_number, slot.index))
            else:
                miss_list.append(brick_key)

        fill_plan: list[tuple[BlockKey3D, TileSlot]] = []
        n_evictions = 0

        for brick_key in miss_list:
            if self.free_slots:
                slot_idx = self.free_slots.pop()
            else:
                slot_idx = self._evict_lru()
                n_evictions += 1

            grid_pos = self._slot_grid_pos(slot_idx)
            slot = TileSlot(index=slot_idx, grid_pos=grid_pos, timestamp=frame_number)

            # Reserve: mark as in-flight only.
            # slot_index is NOT updated here -- the slot is occupied but
            # not yet part of the committed tilemap.
            self._in_flight[slot_idx] = brick_key

            fill_plan.append((brick_key, slot))

        if _CACHE_LOGGER.isEnabledFor(logging.INFO):
            n_hits = len(required_bricks) - len(miss_list)
            n_occupied = len(self.tilemap)
            n_total = self.cache_parameters.n_slots
            _CACHE_LOGGER.info(
                "cache_state  frame=%d  occupied=%d/%d  free=%d  "
                "hits=%d  misses=%d  evictions=%d",
                frame_number,
                n_occupied,
                n_total,
                len(self.free_slots),
                n_hits,
                len(miss_list),
                n_evictions,
            )

        return fill_plan

    def commit(self, brick_key: BlockKey3D, slot: TileSlot) -> None:
        """Move a brick from in-flight to committed (renderable).

        Must be called after ``commit_block`` has written valid data into
        the GPU cache slot.  After this call the brick appears in
        ``tilemap`` and will be picked up by the next ``rebuild_lut``.

        Parameters
        ----------
        brick_key :
            The brick being committed.
        slot :
            The ``TileSlot`` allocated by ``stage()`` for this brick.
        """
        self._in_flight.pop(slot.index, None)
        self.tilemap[brick_key] = slot
        self.slot_index[slot.index] = brick_key
        heapq.heappush(self._lru_heap, (slot.timestamp, slot.index))

    def release_all_in_flight(self) -> None:
        """Return all in-flight slots to the free list.

        Call this immediately after cancelling the ``AsyncSlicer`` task
        for the current update cycle.  Any slots reserved by ``stage()``
        but not yet committed are cleanly reclaimed, so the next
        ``stage()`` call can re-allocate them without leaking slots.

        ``tilemap`` is not touched -- only committed (valid) bricks remain
        renderable after this call.
        """
        for slot_idx in self._in_flight:
            self.free_slots.append(slot_idx)
        self._in_flight.clear()

    # -- Internal helpers ------------------------------------------------

    def _slot_grid_pos(self, flat_idx: int) -> tuple[int, int, int]:
        """Convert flat slot index to 3D cache-grid position ``(sz, sy, sx)``."""
        gs = self.cache_parameters.grid_side
        sz, rem = divmod(flat_idx, gs * gs)
        sy, sx = divmod(rem, gs)
        return (sz, sy, sx)

    def _evict_lru(self) -> int:
        """Evict the least-recently-used *committed* slot.

        Pops heap entries, skipping stale ones, until a valid LRU victim
        is found.  Removes it from ``tilemap`` and ``slot_index`` and
        returns its flat index for reuse.

        In-flight slots are never pushed onto the heap, so they are never
        evicted.

        Complexity: O(log N) amortised.
        """
        while self._lru_heap:
            ts, slot_idx = heapq.heappop(self._lru_heap)

            brick_key = self.slot_index.get(slot_idx)
            if brick_key is None:
                # Free or in-flight slot -- stale heap entry.
                continue

            slot = self.tilemap.get(brick_key)
            if slot is None:
                # Already evicted -- stale heap entry.
                continue

            if slot.timestamp != ts:
                # A later hit refreshed the timestamp -- stale heap entry.
                continue

            # Valid LRU victim -- evict it.
            del self.tilemap[brick_key]
            self.slot_index[slot_idx] = None
            _CACHE_LOGGER.debug("evict  victim=%s  slot=%d", brick_key, slot_idx)
            return slot_idx

        raise RuntimeError("_evict_lru: heap exhausted with no valid victim")

    def clear(self) -> None:
        """Reset to an empty cache, discarding all committed and in-flight bricks."""
        was_occupied = len(self.tilemap)
        self.tilemap.clear()
        self._in_flight.clear()
        for i in range(self.cache_parameters.n_slots):
            self.slot_index[i] = None
        self.slot_index[0] = BlockKey3D(level=0, gz=0, gy=0, gx=0)
        self.free_slots = list(range(self.cache_parameters.n_slots - 1, 0, -1))
        self._lru_heap.clear()
        _CACHE_LOGGER.info("cache_cleared  was_occupied=%d", was_occupied)
