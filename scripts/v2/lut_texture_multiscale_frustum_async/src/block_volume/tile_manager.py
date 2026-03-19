"""Tile manager with LRU eviction for the brick cache.

Phase 3 (optimised): replaces the O(N) linear scan in ``_evict_lru`` with
a min-heap keyed on ``(timestamp, slot_index)``.

LRU eviction with a min-heap
-----------------------------
A min-heap keeps the entry with the smallest timestamp at index 0, so
finding the LRU victim is O(1) and removing it is O(log N).  This reduces
``stage()`` from O(misses × N) to O(misses × log N) — for a 5832-slot
cache that is ~850× fewer comparisons per eviction.

Lazy deletion
-------------
Cache hits update a slot's timestamp in ``tilemap`` but do not update the
heap, because finding and fixing the heap entry would itself be O(N).
Instead the heap is allowed to hold **stale entries** — entries whose
recorded timestamp no longer matches the slot's current timestamp in
``tilemap``.  When ``_evict_lru`` pops the minimum, it checks whether the
entry is stale (the slot's current timestamp is higher than the heap
records) and if so discards it and pops again.  This "lazy" approach keeps
all operations O(log N) amortised, because each entry is pushed at most
once per access and popped at most once.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from block_volume.cache import CacheInfo


@dataclass(frozen=True)
class BrickKey:
    """Identifier for a brick at a specific LOD level.

    Attributes
    ----------
    level : int
        1-indexed LOD level (1 = finest).
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
    """Manages brick-to-slot mapping with LRU eviction.

    Uses a min-heap with lazy deletion for O(log N) eviction instead of
    the previous O(N) linear scan.

    Parameters
    ----------
    cache_info : CacheInfo
        Cache sizing metadata (grid dimensions, slot count, etc.).
    """

    def __init__(self, cache_info: CacheInfo) -> None:
        self.cache_info = cache_info

        # brick → slot
        self.tilemap: dict[BrickKey, TileSlot] = {}
        # slot index → brick (or None)
        self.slot_index: dict[int, BrickKey | None] = {
            i: None for i in range(cache_info.n_slots)
        }
        # Slot 0 is reserved.
        self.slot_index[0] = BrickKey(level=0, gz=0, gy=0, gx=0)
        # Free slots (everything except slot 0).
        self.free_slots: list[int] = list(range(cache_info.n_slots - 1, 0, -1))

        # Min-heap of (timestamp, slot_index) tuples.
        # Invariant: every occupied slot appears at least once in the heap,
        # though some entries may be stale (timestamp < slot.timestamp).
        # Stale entries are discarded lazily in _evict_lru.
        self._lru_heap: list[tuple[int, int]] = []

    def _slot_grid_pos(self, flat_idx: int) -> tuple[int, int, int]:
        """Convert flat slot index to 3D grid position."""
        gs = self.cache_info.grid_side
        sz, rem = divmod(flat_idx, gs * gs)
        sy, sx = divmod(rem, gs)
        return (sz, sy, sx)

    def stage(
        self,
        required_bricks: dict[BrickKey, int],
        frame_number: int,
    ) -> list[tuple[BrickKey, TileSlot]]:
        """Process required bricks: mark hits, plan fills for misses.

        Parameters
        ----------
        required_bricks : dict[BrickKey, int]
            Mapping from brick key to desired level (the level is also
            encoded in the key itself).
        frame_number : int
            Current frame number for LRU timestamps.

        Returns
        -------
        fill_plan : list[tuple[BrickKey, TileSlot]]
            Bricks that need data uploaded, paired with their target slots.
        """
        miss_list: list[BrickKey] = []

        for brick_key in required_bricks:
            if brick_key in self.tilemap:
                # Hit — update timestamp.
                # Push a fresh heap entry; the old one becomes stale and
                # will be skipped lazily when it surfaces in _evict_lru.
                slot = self.tilemap[brick_key]
                slot.timestamp = frame_number
                heapq.heappush(self._lru_heap, (frame_number, slot.index))
            else:
                miss_list.append(brick_key)

        fill_plan: list[tuple[BrickKey, TileSlot]] = []

        for brick_key in miss_list:
            if self.free_slots:
                slot_idx = self.free_slots.pop()
            else:
                # Evict LRU — O(log N) amortised.
                slot_idx = self._evict_lru()

            grid_pos = self._slot_grid_pos(slot_idx)
            slot = TileSlot(index=slot_idx, grid_pos=grid_pos, timestamp=frame_number)

            # Register in maps and push onto the heap.
            self.tilemap[brick_key] = slot
            self.slot_index[slot_idx] = brick_key
            heapq.heappush(self._lru_heap, (frame_number, slot_idx))

            fill_plan.append((brick_key, slot))

        return fill_plan

    def _evict_lru(self) -> int:
        """Evict the least-recently-used slot and return its index.

        Pops entries from the min-heap, discarding stale ones, until a
        valid (non-stale) LRU entry is found.  A heap entry
        ``(ts, slot_idx)`` is stale when the slot's current timestamp in
        ``tilemap`` is higher than ``ts`` (a later hit refreshed it).

        Complexity: O(log N) amortised — each entry is pushed once and
        popped once.
        """
        while self._lru_heap:
            ts, slot_idx = heapq.heappop(self._lru_heap)

            # Look up the brick currently occupying this slot.
            brick_key = self.slot_index.get(slot_idx)
            if brick_key is None:
                # Slot was freed or is the reserved slot 0 — skip.
                continue

            slot = self.tilemap.get(brick_key)
            if slot is None:
                # Brick was already evicted — stale entry.
                continue

            if slot.timestamp != ts:
                # A later hit updated the timestamp — stale entry.
                continue

            # Valid LRU victim found — evict it.
            del self.tilemap[brick_key]
            self.slot_index[slot_idx] = None
            return slot_idx

        raise RuntimeError("_evict_lru: heap exhausted with no valid victim")

    def clear(self) -> None:
        """Remove all resident bricks (reset to empty cache)."""
        self.tilemap.clear()
        for i in range(self.cache_info.n_slots):
            self.slot_index[i] = None
        self.slot_index[0] = BrickKey(level=0, gz=0, gy=0, gx=0)
        self.free_slots = list(range(self.cache_info.n_slots - 1, 0, -1))
        self._lru_heap.clear()
