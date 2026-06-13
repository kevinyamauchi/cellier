"""Tile manager with LRU eviction, late tilemap insertion, and reserve tier.

Key design principle: ``tilemap`` is a *committed* state only.
``stage()`` allocates slots but does **not** insert them into ``tilemap``
— it puts them in ``_in_flight`` instead.  Only ``commit()`` moves a
brick from ``_in_flight`` into ``tilemap``, and only after the data has
been written to the GPU cache.  This means:

* ``rebuild_lut()`` only ever sees bricks with valid data.
* Cancelling an async load is clean: call ``release_all_in_flight()``
  and the reserved-but-never-written slots are returned to the free list
  without touching ``tilemap``.

Reserve tier
------------
When the LOD plan changes (e.g. on zoom-out), bricks that were hot but
are no longer required are not immediately evicted.  Instead they are
placed into ``_reserve``: their GPU texture data is retained but their
LUT entry is removed, so they no longer render.  If the same brick is
re-planned (e.g. user zooms back in), it is promoted from ``_reserve``
to ``tilemap`` with zero network cost.

Demotion is deferred: hot bricks are only moved to ``_reserve`` once
every brick in the new plan has committed (``_pending_plan_count``
reaches zero), so the old bricks continue to serve as a visual
placeholder while the replacement data is loading.

LRU eviction with a min-heap
-----------------------------
A min-heap keeps the entry with the smallest timestamp at index 0, so
finding the LRU victim is O(1) and removing it is O(log N).  This reduces
``stage()`` from O(misses x N) to O(misses x log N).

Reserve bricks are always evicted before hot bricks (``_reserve_lru_heap``
is drained first in ``_evict_lru``), so cache pressure never displaces a
rendered brick when warm data is still available.

Lazy deletion
-------------
Cache hits update a slot's timestamp in ``tilemap`` but do not update the
heap, because finding and fixing the heap entry would itself be O(N).
Instead the heap holds **stale entries** whose recorded timestamp no
longer matches the slot's current timestamp.  ``_evict_lru`` discards
stale entries lazily when it pops them.  Each entry is pushed once and
popped once, so all operations are O(log N) amortised.  The same lazy
deletion applies to ``_reserve_lru_heap``.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cellier.logging import _CACHE_LOGGER

if TYPE_CHECKING:
    from cellier.render.block_cache._cache_parameters_3d import (
        BlockCacheParameters3D,
    )


@dataclass(frozen=True)
class BlockKey3D:
    """Identifier for a brick at a specific LOAD level and slice position.

    Attributes
    ----------
    level : int
        1-indexed LOAD level (1 = finest).
    g0, g1, g2 : int
        Grid position at this level's resolution.  Indexed in the same
        order as the visual's ``displayed_axes`` -- i.e. ``g0`` is the
        brick coordinate along ``displayed_axes[0]``, etc.  For the
        current 3D case with ``displayed_axes=(0, 1, 2)`` these are
        the grid positions along data axes z, y, x respectively.
    slice_coord : tuple of (axis_index, world_value) pairs
        Sorted tuple encoding the non-displayed axis positions at the
        time this brick was requested.  Bricks from different slice
        positions will have distinct keys, allowing the cache to hold
        bricks from multiple slices simultaneously during a transition.
        Empty for purely 3-D data where all axes are displayed.
    """

    level: int
    g0: int
    g1: int
    g2: int
    slice_coord: tuple[tuple[int, int], ...] = ()


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
    brick_max: float = 0.0


class TileManager3D:
    """Manages brick-to-slot mapping with LRU eviction and late insertion.

    Slot lifecycle
    --------------
    1. ``stage()``                 -- allocates a slot; records it in
                                      ``_in_flight`` (NOT in ``tilemap``).
    2. ``commit()``                -- called after data is written to the
                                      GPU cache; moves the entry from
                                      ``_in_flight`` into ``tilemap`` and
                                      pushes it onto the LRU heap.  When
                                      the last outstanding load commits,
                                      ``_flush_pending_demote()`` fires
                                      automatically.
    3. ``release_all_in_flight()`` -- called on cancellation; returns all
                                      reserved-but-not-yet-committed slots
                                      to ``free_slots``.  ``tilemap`` is
                                      untouched, so only valid bricks
                                      remain renderable.

    Reserve tier
    ------------
    Bricks in ``_reserve`` have valid GPU texture data but no LUT entry
    (they are not in ``tilemap``).  They can be promoted back to
    ``tilemap`` instantly (no network fetch) when re-planned.  Under
    memory pressure they are evicted before hot bricks.

    Parameters
    ----------
    cache_parameters :
        Cache sizing metadata (grid dimensions, slot count, etc.).
    """

    def __init__(self, cache_parameters: BlockCacheParameters3D) -> None:
        self.cache_parameters = cache_parameters

        # brick -> slot  (committed, renderable bricks only)
        self.tilemap: dict[BlockKey3D, TileSlot] = {}

        # brick -> slot  (GPU data present, NOT rendered, NOT in LUT)
        self._reserve: dict[BlockKey3D, TileSlot] = {}

        # slot index -> brick  (hot or reserve; None = free or in-flight)
        self.slot_index: dict[int, BlockKey3D | None] = {
            i: None for i in range(cache_parameters.n_slots)
        }
        # Slot 0 is reserved (samples as black / out-of-bounds).
        self.slot_index[0] = BlockKey3D(level=0, g0=0, g1=0, g2=0)

        # Free slots (everything except slot 0).
        self.free_slots: list[int] = list(range(cache_parameters.n_slots - 1, 0, -1))

        # slot index -> brick  (allocated by stage() but not yet committed)
        # These slots are occupied (not in free_slots) but not renderable.
        self._in_flight: dict[int, BlockKey3D] = {}

        # Min-heap of (timestamp, slot_index) for hot LRU eviction.
        # Only committed (hot) slots are pushed onto this heap.
        self._lru_heap: list[tuple[int, int]] = []

        # Min-heap of (timestamp, slot_index) for reserve LRU eviction.
        self._reserve_lru_heap: list[tuple[int, int]] = []

        # Hot bricks to demote to reserve once the current plan fully loads.
        self._pending_demote: set[BlockKey3D] = set()

        # Number of outstanding loads (misses) for the current plan.
        self._pending_plan_count: int = 0

    # -- Slot lifecycle --------------------------------------------------

    def stage(
        self,
        required_bricks: dict[BlockKey3D, int],
        frame_number: int,
    ) -> list[tuple[BlockKey3D, TileSlot]]:
        """Process required bricks: mark hits, promote reserve hits, queue misses.

        Hot hits refresh their LRU timestamp.  Reserve hits are promoted
        to hot instantly (no GPU load needed).  Misses allocate a slot
        (evicting LRU if necessary) and are recorded in ``_in_flight``.

        Bricks currently in ``tilemap`` but absent from ``required_bricks``
        are recorded in ``_pending_demote`` and moved to ``_reserve`` once
        every miss in this plan has committed.  Until then they remain hot
        so they continue to render as a placeholder while new data loads.

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
        required_keys = set(required_bricks.keys())

        # Compute which hot bricks should eventually be demoted.
        # The level-0 sentinel slot is excluded.
        sentinel = BlockKey3D(level=0, g0=0, g1=0, g2=0)
        self._pending_demote = set(self.tilemap.keys()) - required_keys
        self._pending_demote.discard(sentinel)

        miss_list: list[BlockKey3D] = []
        n_promoted = 0

        for brick_key in required_bricks:
            if brick_key in self.tilemap:
                # Hot hit -- refresh timestamp; push a new heap entry (the
                # old one becomes stale and is discarded lazily in _evict_lru).
                slot = self.tilemap[brick_key]
                slot.timestamp = frame_number
                heapq.heappush(self._lru_heap, (frame_number, slot.index))
            elif brick_key in self._reserve:
                # Reserve hit -- promote to hot; no GPU load needed.
                slot = self._reserve.pop(brick_key)
                slot.timestamp = frame_number
                self.tilemap[brick_key] = slot
                self.slot_index[slot.index] = brick_key
                heapq.heappush(self._lru_heap, (frame_number, slot.index))
                n_promoted += 1
            else:
                miss_list.append(brick_key)

        # Track outstanding loads; flush immediately if the plan is already
        # fully satisfied (all hits / warm promotions, no network fetches needed).
        self._pending_plan_count = len(miss_list)
        if self._pending_plan_count == 0:
            self._flush_pending_demote()

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

            # Mark as in-flight only -- slot_index not updated until commit().
            self._in_flight[slot_idx] = brick_key

            fill_plan.append((brick_key, slot))

        if _CACHE_LOGGER.isEnabledFor(logging.INFO):
            n_hot_hits = len(required_bricks) - len(miss_list) - n_promoted
            _CACHE_LOGGER.info(
                "cache_state  frame=%d  hot=%d  reserve=%d  free=%d  "
                "hot_hits=%d  reserve_hits=%d  misses=%d  evictions=%d  "
                "pending_demote=%d",
                frame_number,
                len(self.tilemap),
                len(self._reserve),
                len(self.free_slots),
                n_hot_hits,
                n_promoted,
                len(miss_list),
                n_evictions,
                len(self._pending_demote),
            )

        return fill_plan

    def commit(self, brick_key: BlockKey3D, slot: TileSlot) -> None:
        """Move a brick from in-flight to committed (renderable).

        Must be called after ``commit_block`` has written valid data into
        the GPU cache slot.  After this call the brick appears in
        ``tilemap`` and will be picked up by the next ``rebuild_lut``.

        When this is the last outstanding load for the current plan
        (``_pending_plan_count`` reaches zero), ``_flush_pending_demote``
        fires automatically so the next ``rebuild_lut`` reflects the
        correct LOD.

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

        self._pending_plan_count = max(0, self._pending_plan_count - 1)
        if self._pending_plan_count == 0 and self._pending_demote:
            self._flush_pending_demote()

    def release_all_in_flight(self) -> None:
        """Return all in-flight slots to the free list.

        Call this immediately after cancelling the ``AsyncSlicer`` task
        for the current update cycle.  Any slots reserved by ``stage()``
        but not yet committed are cleanly reclaimed, so the next
        ``stage()`` call can re-allocate them without leaking slots.

        ``tilemap`` is not touched -- only committed (valid) bricks remain
        renderable after this call.

        ``_pending_plan_count`` and ``_pending_demote`` are reset so that
        a subsequent ``stage()`` starts from a clean accounting state.
        Without this reset, a cancel followed by a plan whose miss count
        happens to equal the old stale ``_pending_plan_count`` would
        cause ``_flush_pending_demote`` to fire at the wrong time.
        """
        for slot_idx in self._in_flight:
            self.free_slots.append(slot_idx)
        self._in_flight.clear()
        self._pending_plan_count = 0
        self._pending_demote.clear()

    # -- Internal helpers ------------------------------------------------

    def _slot_grid_pos(self, flat_idx: int) -> tuple[int, int, int]:
        """Convert flat slot index to 3D cache-grid position ``(sz, sy, sx)``."""
        gs = self.cache_parameters.grid_side
        sz, rem = divmod(flat_idx, gs * gs)
        sy, sx = divmod(rem, gs)
        return (sz, sy, sx)

    def _flush_pending_demote(self) -> int:
        """Move ``_pending_demote`` bricks from ``tilemap`` to ``_reserve``.

        Called automatically when ``_pending_plan_count`` reaches zero
        (either synchronously in ``stage()`` for all-hit plans, or from
        the last ``commit()`` of the current plan).

        The caller (``on_data_ready``) triggers ``rebuild_lut()`` after
        all commits, so the LUT will be updated on the next render.

        Returns the number of bricks demoted.
        """
        n = 0
        for key in self._pending_demote:
            slot = self.tilemap.pop(key, None)
            if slot is None:
                continue
            self._reserve[key] = slot
            # slot_index retains the mapping — slot is still occupied.
            heapq.heappush(self._reserve_lru_heap, (slot.timestamp, slot.index))
            n += 1
        self._pending_demote.clear()
        if n:
            _CACHE_LOGGER.debug(
                "demote_to_reserve  count=%d  reserve_size=%d", n, len(self._reserve)
            )
        return n

    def _evict_lru(self) -> int:
        """Evict the least-recently-used slot, preferring reserve over hot.

        Reserve bricks are drained first (no visual disruption).  Only
        when the reserve is exhausted does this fall back to evicting hot
        (rendered) bricks.

        Pops heap entries, skipping stale ones, until a valid LRU victim
        is found.  Removes it from ``_reserve`` or ``tilemap`` and
        ``slot_index`` and returns the flat index for reuse.

        In-flight slots are never pushed onto either heap and are never
        evicted.

        Complexity: O(log N) amortised.
        """
        # Prefer evicting reserve bricks -- no rendered brick is displaced.
        while self._reserve_lru_heap:
            ts, slot_idx = heapq.heappop(self._reserve_lru_heap)

            brick_key = self.slot_index.get(slot_idx)
            if brick_key is None:
                continue

            slot = self._reserve.get(brick_key)
            if slot is None:
                # Already evicted from reserve (promoted or evicted earlier).
                continue

            if slot.timestamp != ts:
                # A later promotion refreshed the timestamp -- stale entry.
                continue

            del self._reserve[brick_key]
            self.slot_index[slot_idx] = None
            _CACHE_LOGGER.debug(
                "evict_reserve  victim=%s  slot=%d", brick_key, slot_idx
            )
            return slot_idx

        # Fall back to hot (rendered) LRU eviction.
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
            _CACHE_LOGGER.debug("evict_hot  victim=%s  slot=%d", brick_key, slot_idx)
            return slot_idx

        raise RuntimeError("_evict_lru: heap exhausted with no valid victim")

    def evict_finer_than(self, min_level: int) -> int:
        """Evict all committed bricks with level < min_level.

        Removes entries from ``tilemap`` and returns their slots to
        ``free_slots``.  In-flight slots are unaffected (they are handled
        by ``release_all_in_flight``).

        In the 3D pipeline this is useful when ``force_level`` is set: all
        bricks are requested at a single level, so any finer bricks left
        over from a previous frame are not needed and can be reclaimed
        proactively rather than waiting for LRU eviction.

        Parameters
        ----------
        min_level : int
            Evict bricks whose level is strictly less than this value.

        Returns
        -------
        int
            Number of bricks evicted.
        """
        to_evict = [key for key in self.tilemap if key.level < min_level]
        for key in to_evict:
            slot = self.tilemap.pop(key)
            self.slot_index[slot.index] = None
            self.free_slots.append(slot.index)
        if to_evict:
            _CACHE_LOGGER.debug(
                "evict_finer_than  min_level=%d  evicted=%d", min_level, len(to_evict)
            )
        return len(to_evict)

    def clear(self) -> None:
        """Reset to an empty cache, discarding all committed and in-flight bricks."""
        was_occupied = len(self.tilemap)
        was_reserve = len(self._reserve)
        self.tilemap.clear()
        self._reserve.clear()
        self._in_flight.clear()
        for i in range(self.cache_parameters.n_slots):
            self.slot_index[i] = None
        self.slot_index[0] = BlockKey3D(level=0, g0=0, g1=0, g2=0)
        self.free_slots = list(range(self.cache_parameters.n_slots - 1, 0, -1))
        self._lru_heap.clear()
        self._reserve_lru_heap.clear()
        self._pending_demote.clear()
        self._pending_plan_count = 0
        _CACHE_LOGGER.info(
            "cache_cleared  was_occupied=%d  was_reserve=%d",
            was_occupied,
            was_reserve,
        )
