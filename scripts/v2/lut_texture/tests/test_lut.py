import numpy as np

from block_volume.layout import BlockLayout
from block_volume.lut import build_identity_lut


class TestBuildIdentityLut:
    def test_shape(self):
        layout = BlockLayout.from_volume_shape((64, 64, 64), block_size=32)
        tex = build_identity_lut(layout)
        assert tex.data.shape == (2, 2, 2, 4)
        assert tex.data.dtype == np.uint8

    def test_identity_mapping(self):
        layout = BlockLayout.from_volume_shape((96, 64, 128), block_size=32)
        tex = build_identity_lut(layout)
        gd, gh, gw = layout.grid_dims

        for d in range(gd):
            for h in range(gh):
                for w in range(gw):
                    # tile_x = w, tile_y = h, tile_z = d, level = 1
                    expected = [w, h, d, 1]
                    actual = list(tex.data[d, h, w])
                    assert actual == expected, (
                        f"Mismatch at ({d},{h},{w}): "
                        f"expected {expected}, got {actual}"
                    )

    def test_level_is_one(self):
        layout = BlockLayout.from_volume_shape((64, 64, 64), block_size=32)
        tex = build_identity_lut(layout)
        np.testing.assert_array_equal(tex.data[..., 3], 1)

    def test_format_is_rgba8uint(self):
        layout = BlockLayout.from_volume_shape((64, 64, 64), block_size=32)
        tex = build_identity_lut(layout)
        assert tex.format == "rgba8uint"

    def test_non_square_grid(self):
        layout = BlockLayout.from_volume_shape((32, 64, 96), block_size=32)
        tex = build_identity_lut(layout)
        assert tex.data.shape == (1, 2, 3, 4)
        # Corner cases
        assert list(tex.data[0, 0, 0]) == [0, 0, 0, 1]
        assert list(tex.data[0, 1, 2]) == [2, 1, 0, 1]
