from cellier.v2.render.block_cache._cache_parameters_2d import (
    BlockCacheParameters2D,
    build_cache_texture_2d,
)
from cellier.v2.render.block_cache._tile_manager_2d import TileManager2D


class BlockCache2D:
    """Fixed-size GPU slot pool with LRU eviction.

    Parameters
    ----------
    cache_parameters : CacheInfo
        Cache sizing metadata produced by ``compute_cache_info()``.

    Attributes
    ----------
    cache_parameters : BlockCacheParameters2D
        Cache sizing metadata (grid dims, slot count, padded brick size).
    tile_manager : TileManager
        Brick-to-slot mapping with LRU eviction.
    cache_data : np.ndarray
        CPU-side backing array, shape ``(cD, cH, cW)``, dtype float32.
    cache_tex : gfx.Texture
        GPU 3-D float32 texture wrapping ``cache_data``.
    """

    def __init__(self, cache_parameters: BlockCacheParameters2D) -> None:
        self.info = cache_parameters
        self.tile_manager = TileManager2D(cache_parameters)
        self.cache_data, self.cache_tex = build_cache_texture_2d(cache_parameters)
