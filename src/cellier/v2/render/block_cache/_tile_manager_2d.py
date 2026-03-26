"""Tile manager with LRU eviction for the 2D tile cache.

Manages the bidirectional map BlockKey2D <-> TileSlot with O(log N)
LRU eviction via a min-heap with lazy deletion.

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
                                  untouched, so only valid tiles remain
                                  renderable.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cellier.v2.render.block_cache._cache_parameters_2d import (
        BlockCacheParameters2D,
    )


@dataclass(frozen=True)
class BlockKey2D:
    """Identifier for a tile at a specific LOD level.

    Attributes
    ----------
    level : int
        1-indexed LOD level (1 = finest).
    gy, gx : int
        Grid position at this level's resolution.
    """

    level: int
    gy: int
    gx: int


@dataclass
class TileSlot:
    """A slot in the GPU cache.

    Attributes
    ----------
    index : int
        Flat slot index (0 = reserved, never used for data).
    grid_pos : tuple[int, int]
        ``(sy, sx)`` in the cache grid.
    timestamp : int
        Frame number when last accessed (for LRU).
    """

    index: int
    grid_pos: tuple[int, int]
    timestamp: int = 0


class TileManager2D:
    """Manage tile-to-slot mapping with LRU eviction and late insertion.

    Uses a min-heap with lazy deletion for O(log N) eviction.

    Parameters
    ----------
    cache_parameters : BlockCacheParameters2D
        Cache sizing metadata (grid dimensions, slot count, etc.).
    """

    def __init__(self, cache_parameters: BlockCacheParameters2D) -> None:
        self.cache_info = cache_parameters

        # tile -> slot  (committed, renderable tiles only)
        self.tilemap: dict[BlockKey2D, TileSlot] = {}
        # slot index -> tile  (committed tiles only; None = free or in-flight)
        self.slot_index: dict[int, BlockKey2D | None] = {
            i: None for i in range(cache_parameters.n_slots)
        }
        # Slot 0 is reserved (empty/black).
        self.slot_index[0] = BlockKey2D(level=0, gy=0, gx=0)
        # Free slots (everything except slot 0).
        self.free_slots: list[int] = list(range(cache_parameters.n_slots - 1, 0, -1))

        # slot index -> tile  (allocated by stage() but not yet committed)
        self._in_flight: dict[int, BlockKey2D] = {}

        # Min-heap of (timestamp, slot_index) tuples.
        self._lru_heap: list[tuple[int, int]] = []

    def _slot_grid_pos(self, flat_idx: int) -> tuple[int, int]:
        """Convert flat slot index to 2D grid position ``(sy, sx)``."""
        gs = self.cache_info.grid_side
        sy, sx = divmod(flat_idx, gs)
        return (sy, sx)

    def stage(
        self,
        required: dict[BlockKey2D, int],
        frame_number: int,
    ) -> list[tuple[BlockKey2D, TileSlot]]:
        """Process required tiles: mark hits, reserve slots for misses.

        Hits update the timestamp of the existing committed slot.
        Misses allocate a slot (evicting LRU if necessary) and record it
        in ``_in_flight``.  The slot is **not** added to ``tilemap`` here
        -- only ``commit()`` does that.

        Parameters
        ----------
        required : dict[BlockKey2D, int]
            Mapping from tile key to desired level.
        frame_number : int
            Current frame number for LRU timestamps.

        Returns
        -------
        fill_plan : list[tuple[BlockKey2D, TileSlot]]
            Tiles that need data uploaded, paired with their target slots.
        """
        miss_list: list[BlockKey2D] = []

        for tile_key in required:
            if tile_key in self.tilemap:
                # Hit -- refresh timestamp.
                slot = self.tilemap[tile_key]
                slot.timestamp = frame_number
                heapq.heappush(self._lru_heap, (frame_number, slot.index))
            else:
                miss_list.append(tile_key)

        fill_plan: list[tuple[BlockKey2D, TileSlot]] = []

        for tile_key in miss_list:
            slot_idx = self.free_slots.pop() if self.free_slots else self._evict_lru()

            grid_pos = self._slot_grid_pos(slot_idx)
            slot = TileSlot(index=slot_idx, grid_pos=grid_pos, timestamp=frame_number)

            # Reserve: mark as in-flight only.
            self._in_flight[slot_idx] = tile_key

            fill_plan.append((tile_key, slot))

        return fill_plan

    def commit(self, tile_key: BlockKey2D, slot: TileSlot) -> None:
        """Move a tile from in-flight to committed (renderable).

        Must be called after data has been written into the GPU cache
        slot.  After this call the tile appears in ``tilemap`` and will
        be picked up by the next ``rebuild_lut``.

        Parameters
        ----------
        tile_key : BlockKey2D
            The tile being committed.
        slot : TileSlot
            The slot returned by ``stage()`` for this tile.
        """
        self._in_flight.pop(slot.index, None)
        self.tilemap[tile_key] = slot
        self.slot_index[slot.index] = tile_key
        heapq.heappush(self._lru_heap, (slot.timestamp, slot.index))

    def release_all_in_flight(self) -> None:
        """Return all reserved-but-not-committed slots to the free pool.

        Called on cancellation so that slots reserved by ``stage()`` but
        never committed are reclaimed.  ``tilemap`` is untouched, so only
        valid tiles remain renderable.
        """
        for slot_idx in list(self._in_flight.keys()):
            self.free_slots.append(slot_idx)
        self._in_flight.clear()

    def _evict_lru(self) -> int:
        """Evict the least-recently-used slot and return its index.

        Pops entries from the min-heap, discarding stale ones, until a
        valid (non-stale) LRU entry is found.
        """
        while self._lru_heap:
            ts, slot_idx = heapq.heappop(self._lru_heap)

            tile_key = self.slot_index.get(slot_idx)
            if tile_key is None:
                continue

            slot = self.tilemap.get(tile_key)
            if slot is None:
                continue

            if slot.timestamp != ts:
                # Stale -- a later hit refreshed it.
                continue

            # Valid LRU victim found.
            del self.tilemap[tile_key]
            self.slot_index[slot_idx] = None
            return slot_idx

        raise RuntimeError("_evict_lru: heap exhausted with no valid victim")

    def clear(self) -> None:
        """Remove all resident tiles (reset to empty cache)."""
        self.tilemap.clear()
        self._in_flight.clear()
        for i in range(self.cache_info.n_slots):
            self.slot_index[i] = None
        self.slot_index[0] = BlockKey2D(level=0, gy=0, gx=0)
        self.free_slots = list(range(self.cache_info.n_slots - 1, 0, -1))
        self._lru_heap.clear()
