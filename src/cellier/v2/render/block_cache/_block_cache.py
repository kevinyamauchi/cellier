"""Generic GPU brick-cache for 3-D volume rendering.

Manages a fixed-size 3-D texture that acts as a flat pool of brick
slots, and the TileManager that tracks which logical brick occupies
which physical slot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cellier.v2.logging import _GPU_LOGGER
from cellier.v2.render.block_cache._cache_parameters_3d import (
    BlockCacheParameters3D,
    build_cache_texture_3d,
    commit_block_3d,
)
from cellier.v2.render.block_cache._tile_manager import (
    BlockKey3D,
    TileManager,
    TileSlot,
)

if TYPE_CHECKING:
    import numpy as np


class BlockCache3D:
    """Fixed-size GPU slot pool with LRU eviction.

    Parameters
    ----------
    cache_parameters : CacheInfo
        Cache sizing metadata produced by ``compute_cache_info()``.

    Attributes
    ----------
    info : CacheInfo
        Cache sizing metadata (grid dims, slot count, padded brick size).
    tile_manager : TileManager
        Brick-to-slot mapping with LRU eviction.
    cache_data : np.ndarray
        CPU-side backing array, shape ``(cD, cH, cW)``, dtype float32.
    cache_tex : gfx.Texture
        GPU 3-D float32 texture wrapping ``cache_data``.
    """

    def __init__(self, cache_parameters: BlockCacheParameters3D) -> None:
        self.info = cache_parameters
        self.tile_manager = TileManager(cache_parameters)
        self.cache_data, self.cache_tex = build_cache_texture_3d(cache_parameters)

    def stage(
        self,
        required_bricks: dict[BlockKey3D, int],
        frame_number: int,
    ) -> list[tuple[BlockKey3D, TileSlot]]:
        """Classify required bricks as hits or misses; return fill_plan.

        Cache hits have their LRU timestamp refreshed.  Cache misses are
        allocated a slot — evicting the least-recently-used occupant if
        the cache is full — and returned as a fill_plan for the caller
        to load asynchronously.

        Parameters
        ----------
        required_bricks : dict[BrickKey, int]
            Mapping from brick key to desired LOAD level.  Only the keys
            are used here; the level is already encoded in ``BrickKey``.
        frame_number : int
            Monotonically increasing counter used for LRU timestamps.

        Returns
        -------
        fill_plan : list[tuple[BrickKey, TileSlot]]
            Bricks that need data uploaded, paired with their target slot.
            Empty when every required brick was already resident.
        """
        return self.tile_manager.stage(required_bricks, frame_number)

    def write_brick(
        self, slot: TileSlot, data: np.ndarray, key: BlockKey3D | None = None
    ) -> None:
        """Write a padded brick into the CPU array and mark dirty for GPU upload.

        The actual GPU transfer is deferred until the next
        ``renderer.render()`` call (uses pygfx ``update_range``).

        Parameters
        ----------
        slot : TileSlot
            Target slot — ``grid_pos`` determines the write offset.
        data : np.ndarray
            Float32 array of shape ``(pbs, pbs, pbs)`` where
            ``pbs = block_size + 2 * overlap``.
        key : BlockKey3D or None
            Brick identity for logging.
        """
        commit_block_3d(
            cache_data=self.cache_data,
            cache_tex=self.cache_tex,
            grid_pos=slot.grid_pos,
            padded_block_size=self.info.padded_block_size,
            data=data,
        )
        _GPU_LOGGER.debug(
            "brick_written  key=%s  slot=%d  grid_pos=%s",
            key,
            slot.index,
            slot.grid_pos,
        )

    @property
    def n_resident(self) -> int:
        """Number of bricks currently resident in the cache."""
        return len(self.tile_manager.tilemap)

    def clear(self) -> None:
        """Evict all resident bricks and reset the cache to empty."""
        self.tile_manager.clear()
