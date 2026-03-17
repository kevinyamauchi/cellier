import math

import pytest

from block_volume.layout import BLOCK_SIZE_DEFAULT, BlockLayout


class TestBlockLayoutBasic:
    def test_exact_divisible(self):
        layout = BlockLayout.from_volume_shape((64, 64, 64), block_size=32)
        assert layout.grid_dims == (2, 2, 2)
        assert layout.padded_shape == (64, 64, 64)
        assert layout.n_bricks == 8

    def test_non_divisible_rounds_up(self):
        layout = BlockLayout.from_volume_shape((65, 33, 100), block_size=32)
        assert layout.grid_dims == (
            math.ceil(65 / 32),
            math.ceil(33 / 32),
            math.ceil(100 / 32),
        )
        assert layout.padded_shape == (3 * 32, 2 * 32, 4 * 32)

    def test_n_bricks(self):
        layout = BlockLayout.from_volume_shape((100, 100, 100), block_size=32)
        gd, gh, gw = layout.grid_dims
        assert layout.n_bricks == gd * gh * gw

    def test_default_block_size(self):
        layout = BlockLayout.from_volume_shape((64, 64, 64))
        assert layout.block_size == BLOCK_SIZE_DEFAULT

    def test_small_volume(self):
        layout = BlockLayout.from_volume_shape((1, 1, 1), block_size=32)
        assert layout.grid_dims == (1, 1, 1)
        assert layout.padded_shape == (32, 32, 32)
        assert layout.n_bricks == 1


class TestBlockLayoutValidation:
    def test_grid_dim_exceeds_255_raises(self):
        # block_size=1 on a 256-voxel axis gives grid_dim=256
        with pytest.raises(ValueError, match="exceeds uint8 range"):
            BlockLayout.from_volume_shape((256, 1, 1), block_size=1)

    def test_grid_dim_exactly_255_ok(self):
        layout = BlockLayout.from_volume_shape((255, 1, 1), block_size=1)
        assert layout.grid_dims[0] == 255


class TestBlockLayoutFrozen:
    def test_immutable(self):
        layout = BlockLayout.from_volume_shape((64, 64, 64), block_size=32)
        with pytest.raises(AttributeError):
            layout.block_size = 16
