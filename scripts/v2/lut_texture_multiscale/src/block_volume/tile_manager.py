"""Tile manager with LRU eviction for the brick cache.

Tracks which bricks are resident in the GPU cache and handles
allocation / eviction of cache slots.  All LOD, eviction, and LUT
logic lives here (Python-side); the WGSL shader is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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

    Parameters
    ----------
    cache_info : CacheInfo
        Cache sizing metadata (grid dimensions, slot count, etc.).
    """

    def __init__(self, cache_info: CacheInfo) -> None:
        self.cache_info = cache_info

        # brick -> slot
        self.tilemap: dict[BrickKey, TileSlot] = {}
        # slot index -> brick (or None)
        self.slot_index: dict[int, BrickKey | None] = {
            i: None for i in range(cache_info.n_slots)
        }
        # Slot 0 is reserved.
        self.slot_index[0] = BrickKey(level=0, gz=0, gy=0, gx=0)
        # Free slots (everything except slot 0).
        self.free_slots: list[int] = list(range(cache_info.n_slots - 1, 0, -1))

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
            Mapping from brick key to desired level (unused here, the
            level is already encoded in the key).
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
                self.tilemap[brick_key].timestamp = frame_number
            else:
                miss_list.append(brick_key)

        fill_plan: list[tuple[BrickKey, TileSlot]] = []

        for brick_key in miss_list:
            if self.free_slots:
                slot_idx = self.free_slots.pop()
            else:
                # Evict LRU.
                slot_idx = self._evict_lru()

            grid_pos = self._slot_grid_pos(slot_idx)
            slot = TileSlot(index=slot_idx, grid_pos=grid_pos, timestamp=frame_number)

            # Register in maps.
            self.tilemap[brick_key] = slot
            self.slot_index[slot_idx] = brick_key

            fill_plan.append((brick_key, slot))

        return fill_plan

    def _evict_lru(self) -> int:
        """Evict the least-recently-used slot and return its index."""
        oldest_key: BrickKey | None = None
        oldest_ts = float("inf")

        for key, slot in self.tilemap.items():
            if slot.timestamp < oldest_ts:
                oldest_ts = slot.timestamp
                oldest_key = key

        assert oldest_key is not None, "Cannot evict: tilemap is empty"
        victim_slot = self.tilemap.pop(oldest_key)
        self.slot_index[victim_slot.index] = None
        return victim_slot.index

    def clear(self) -> None:
        """Remove all resident bricks (reset to empty cache)."""
        self.tilemap.clear()
        for i in range(self.cache_info.n_slots):
            self.slot_index[i] = None
        self.slot_index[0] = BrickKey(level=0, gz=0, gy=0, gx=0)
        self.free_slots = list(range(self.cache_info.n_slots - 1, 0, -1))
