"""Tile manager with LRU eviction for the 2D tile cache.

Manages the bidirectional map TileKey <-> TileSlot with O(log N)
LRU eviction via a min-heap with lazy deletion.

This is the 2D specialisation: TileKey omits ``gz`` and
``_slot_grid_pos`` uses 2D grid arithmetic.  A 3D subclass (Phase 2)
would override ``_slot_grid_pos`` to use 3D arithmetic.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from image_block.shared.cache import CacheInfo


@dataclass(frozen=True)
class TileKey:
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


class TileManager:
    """Manage tile-to-slot mapping with LRU eviction.

    Uses a min-heap with lazy deletion for O(log N) eviction.

    Parameters
    ----------
    cache_info : CacheInfo
        Cache sizing metadata (grid dimensions, slot count, etc.).
    """

    def __init__(self, cache_info: CacheInfo) -> None:
        self.cache_info = cache_info

        # tile -> slot
        self.tilemap: dict[TileKey, TileSlot] = {}
        # slot index -> tile (or None)
        self.slot_index: dict[int, TileKey | None] = {
            i: None for i in range(cache_info.n_slots)
        }
        # Slot 0 is reserved (empty/black).
        self.slot_index[0] = TileKey(level=0, gy=0, gx=0)
        # Free slots (everything except slot 0).
        self.free_slots: list[int] = list(
            range(cache_info.n_slots - 1, 0, -1)
        )

        # Min-heap of (timestamp, slot_index) tuples.
        self._lru_heap: list[tuple[int, int]] = []

    def _slot_grid_pos(self, flat_idx: int) -> tuple[int, int]:
        """Convert flat slot index to 2D grid position ``(sy, sx)``."""
        gs = self.cache_info.grid_side
        sy, sx = divmod(flat_idx, gs)
        return (sy, sx)

    def stage(
        self,
        required: dict[TileKey, int],
        frame_number: int,
    ) -> list[tuple[TileKey, TileSlot]]:
        """Process required tiles: mark hits, plan fills for misses.

        Parameters
        ----------
        required : dict[TileKey, int]
            Mapping from tile key to desired level.
        frame_number : int
            Current frame number for LRU timestamps.

        Returns
        -------
        fill_plan : list[tuple[TileKey, TileSlot]]
            Tiles that need data uploaded, paired with their target slots.
        """
        miss_list: list[TileKey] = []

        for tile_key in required:
            if tile_key in self.tilemap:
                # Hit -- update timestamp.
                slot = self.tilemap[tile_key]
                slot.timestamp = frame_number
                heapq.heappush(self._lru_heap, (frame_number, slot.index))
            else:
                miss_list.append(tile_key)

        fill_plan: list[tuple[TileKey, TileSlot]] = []

        for tile_key in miss_list:
            if self.free_slots:
                slot_idx = self.free_slots.pop()
            else:
                slot_idx = self._evict_lru()

            grid_pos = self._slot_grid_pos(slot_idx)
            slot = TileSlot(
                index=slot_idx, grid_pos=grid_pos, timestamp=frame_number
            )

            self.tilemap[tile_key] = slot
            self.slot_index[slot_idx] = tile_key
            heapq.heappush(self._lru_heap, (frame_number, slot_idx))

            fill_plan.append((tile_key, slot))

        return fill_plan

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
        for i in range(self.cache_info.n_slots):
            self.slot_index[i] = None
        self.slot_index[0] = TileKey(level=0, gy=0, gx=0)
        self.free_slots = list(
            range(self.cache_info.n_slots - 1, 0, -1)
        )
        self._lru_heap.clear()


if __name__ == "__main__":
    # Sanity check.
    info = CacheInfo(
        grid_side=4,
        n_slots=16,
        padded_block_size=34,
        overlap=1,
    )
    mgr = TileManager(info)
    print(f"Created TileManager: {info.n_slots} slots, "
          f"grid_side={info.grid_side}")
    print(f"  free_slots: {len(mgr.free_slots)}")
    print(f"  slot 0 reserved for: {mgr.slot_index[0]}")

    # Stage some tiles.
    required = {
        TileKey(1, 0, 0): 1,
        TileKey(1, 0, 1): 1,
        TileKey(1, 1, 0): 1,
    }
    plan = mgr.stage(required, frame_number=1)
    print(f"  staged {len(plan)} fills from {len(required)} required")
    for tk, ts in plan:
        print(f"    {tk} -> slot {ts.index} @ grid {ts.grid_pos}")

    # Re-stage: all should be hits now.
    plan2 = mgr.stage(required, frame_number=2)
    print(f"  re-stage: {len(plan2)} fills (should be 0)")
    print("TileManager OK")
