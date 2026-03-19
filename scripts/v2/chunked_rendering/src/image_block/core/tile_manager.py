"""Tile manager with LRU eviction for 2D and 3D block caches.

Manages the bidirectional map BlockKey <-> TileSlot with O(log N)
LRU eviction via a min-heap with lazy deletion.

Dispatches on ``cache_info.ndim`` for the ``_slot_grid_pos`` calculation.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from image_block.core.block_key import BlockKey
from image_block.core.cache import CacheInfo


@dataclass
class TileSlot:
    """A slot in the GPU cache.

    Attributes
    ----------
    index : int
        Flat slot index (0 = reserved, never used for data).
    grid_pos : tuple[int, ...]
        ``(sy, sx)`` for 2D or ``(sz, sy, sx)`` for 3D in the cache grid.
    timestamp : int
        Frame number when last accessed (for LRU).
    """

    index: int
    grid_pos: tuple[int, ...]
    timestamp: int = 0


class TileManager:
    """Manage block-to-slot mapping with LRU eviction.

    Uses a min-heap with lazy deletion for O(log N) eviction.

    Parameters
    ----------
    cache_info : CacheInfo
        Cache sizing metadata (grid dimensions, slot count, ndim, etc.).
    """

    def __init__(self, cache_info: CacheInfo) -> None:
        self.cache_info = cache_info
        self._ndim = cache_info.ndim

        # block -> slot
        self.tilemap: dict[BlockKey, TileSlot] = {}
        # slot index -> block (or None)
        self.slot_index: dict[int, BlockKey | None] = {
            i: None for i in range(cache_info.n_slots)
        }
        # Slot 0 is reserved (empty/black).
        self.slot_index[0] = BlockKey(level=0, gz=0, gy=0, gx=0)
        # Free slots (everything except slot 0).
        self.free_slots: list[int] = list(range(cache_info.n_slots - 1, 0, -1))

        # Min-heap of (timestamp, slot_index) tuples.
        self._lru_heap: list[tuple[int, int]] = []

    def _slot_grid_pos(self, flat_idx: int) -> tuple[int, ...]:
        """Convert flat slot index to grid position in numpy axis order."""
        gs = self.cache_info.grid_side
        if self._ndim == 2:
            sy, sx = divmod(flat_idx, gs)
            return (sy, sx)
        else:  # ndim == 3
            sz, rem = divmod(flat_idx, gs * gs)
            sy, sx = divmod(rem, gs)
            return (sz, sy, sx)

    def stage(
        self,
        required: dict[BlockKey, int],
        frame_number: int,
    ) -> list[tuple[BlockKey, TileSlot]]:
        """Process required blocks: mark hits, plan fills for misses.

        Parameters
        ----------
        required : dict[BlockKey, int]
            Mapping from block key to desired level.
        frame_number : int
            Current frame number for LRU timestamps.

        Returns
        -------
        fill_plan : list[tuple[BlockKey, TileSlot]]
            Blocks that need data uploaded, paired with their target slots.
        """
        miss_list: list[BlockKey] = []

        for block_key in required:
            if block_key in self.tilemap:
                # Hit -- update timestamp.
                slot = self.tilemap[block_key]
                slot.timestamp = frame_number
                heapq.heappush(self._lru_heap, (frame_number, slot.index))
            else:
                miss_list.append(block_key)

        fill_plan: list[tuple[BlockKey, TileSlot]] = []

        for block_key in miss_list:
            if self.free_slots:
                slot_idx = self.free_slots.pop()
            else:
                slot_idx = self._evict_lru()

            grid_pos = self._slot_grid_pos(slot_idx)
            slot = TileSlot(index=slot_idx, grid_pos=grid_pos, timestamp=frame_number)

            self.tilemap[block_key] = slot
            self.slot_index[slot_idx] = block_key
            heapq.heappush(self._lru_heap, (frame_number, slot_idx))

            fill_plan.append((block_key, slot))

        return fill_plan

    def _evict_lru(self) -> int:
        """Evict the least-recently-used slot and return its index.

        Pops entries from the min-heap, discarding stale ones, until a
        valid (non-stale) LRU entry is found.
        """
        while self._lru_heap:
            ts, slot_idx = heapq.heappop(self._lru_heap)

            block_key = self.slot_index.get(slot_idx)
            if block_key is None:
                continue

            slot = self.tilemap.get(block_key)
            if slot is None:
                continue

            if slot.timestamp != ts:
                # Stale -- a later hit refreshed it.
                continue

            # Valid LRU victim found.
            del self.tilemap[block_key]
            self.slot_index[slot_idx] = None
            return slot_idx

        raise RuntimeError("_evict_lru: heap exhausted with no valid victim")

    def clear(self) -> None:
        """Remove all resident blocks (reset to empty cache)."""
        self.tilemap.clear()
        for i in range(self.cache_info.n_slots):
            self.slot_index[i] = None
        self.slot_index[0] = BlockKey(level=0, gz=0, gy=0, gx=0)
        self.free_slots = list(range(self.cache_info.n_slots - 1, 0, -1))
        self._lru_heap.clear()
