import numpy as np

from block_volume.cache import build_cache_texture
from block_volume.layout import BlockLayout


class TestBuildCacheTexture:
    def test_exact_shape_no_padding(self):
        vol = np.ones((64, 64, 64), dtype=np.float32)
        layout = BlockLayout.from_volume_shape(vol.shape, block_size=32)
        tex = build_cache_texture(vol, layout)
        assert tex.data.shape == (64, 64, 64)
        assert tex.data.dtype == np.float32

    def test_padding_applied(self):
        vol = np.ones((33, 33, 33), dtype=np.float32)
        layout = BlockLayout.from_volume_shape(vol.shape, block_size=32)
        tex = build_cache_texture(vol, layout)
        assert tex.data.shape == layout.padded_shape
        # Original region should be 1.0
        assert tex.data[0, 0, 0] == 1.0
        assert tex.data[32, 32, 32] == 1.0
        # Padded region should be 0.0
        assert tex.data[63, 63, 63] == 0.0

    def test_original_data_preserved(self):
        rng = np.random.default_rng(42)
        vol = rng.random((50, 60, 70), dtype=np.float32)
        layout = BlockLayout.from_volume_shape(vol.shape, block_size=32)
        tex = build_cache_texture(vol, layout)
        np.testing.assert_array_equal(
            tex.data[:50, :60, :70], vol
        )

    def test_dtype_cast_from_float64(self):
        vol = np.ones((32, 32, 32), dtype=np.float64)
        layout = BlockLayout.from_volume_shape(vol.shape, block_size=32)
        tex = build_cache_texture(vol, layout)
        assert tex.data.dtype == np.float32
