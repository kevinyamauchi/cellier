import numpy as np

from cellier.v2.render.block_cache import (
    BlockKey3D,
    TileManager,
    compute_block_cache_parameters,
)
from cellier.v2.render.lut_indirection import BlockLayout3D, LutIndirectionManager3D

CACHE_INFO = compute_block_cache_parameters(block_size=4, gpu_budget_bytes=8 * 6**3 * 4)

# LUT layout: 4x4x4 finest grid (volume 4x4x4, block_size=1), 2 LOD levels.
BASE_LAYOUT = BlockLayout3D(volume_shape=(4, 4, 4), block_size=1)
N_LEVELS = 2


def test_rebuild_empty_lut_is_all_zeros() -> None:
    lut = LutIndirectionManager3D(BASE_LAYOUT, n_levels=N_LEVELS)
    tile_manager = TileManager(CACHE_INFO)
    lut.rebuild(tile_manager)
    assert np.all(lut.lut_data == 0)


def test_coarse_brick_fills_its_fine_cells() -> None:
    """A level-2 brick at (gz=0,gy=0,gx=0) covers [0:2, 0:2, 0:2] of the
    finest grid. After rebuild, every cell in that region should carry
    level=2 and the slot's (sx, sy, sz) coordinates.

    Slot assignment: tile_manager allocates slot 1 first (free_slots pops
    from the end). With grid_side=2, slot 1 has grid_pos=(0, 0, 1), so
    sz=0, sy=0, sx=1 and the LUT channels are (sx=1, sy=0, sz=0, level=2).
    """
    lut = LutIndirectionManager3D(BASE_LAYOUT, n_levels=N_LEVELS)
    tile_manager = TileManager(CACHE_INFO)
    coarse_key = BlockKey3D(level=2, gz=0, gy=0, gx=0)
    fill_plan = tile_manager.stage({coarse_key: 2}, frame_number=1)
    slot = fill_plan[0][1]
    sz, sy, sx = slot.grid_pos

    lut.rebuild(tile_manager)

    covered = lut.lut_data[0:2, 0:2, 0:2]
    assert np.all(covered[..., 0] == sx)
    assert np.all(covered[..., 1] == sy)
    assert np.all(covered[..., 2] == sz)
    assert np.all(covered[..., 3] == 2)


def test_uncovered_cells_remain_zero() -> None:
    """Cells outside the coarse brick's footprint stay at level 0."""
    lut = LutIndirectionManager3D(BASE_LAYOUT, n_levels=N_LEVELS)
    tile_manager = TileManager(CACHE_INFO)
    coarse_key = BlockKey3D(level=2, gz=0, gy=0, gx=0)
    tile_manager.stage({coarse_key: 2}, frame_number=1)

    lut.rebuild(tile_manager)

    assert np.all(lut.lut_data[2:, :, :] == 0)
    assert np.all(lut.lut_data[:, 2:, :] == 0)
    assert np.all(lut.lut_data[:, :, 2:] == 0)


def test_fine_brick_overwrites_coarse_at_its_cell() -> None:
    """With a coarse (level-2) and a fine (level-1) brick both resident:
    - lut[0,0,0] belongs to the fine brick → level=1
    - lut[0,0,1] is covered by coarse only → level=2
    - lut[2,0,0] is covered by neither → level=0
    """
    lut = LutIndirectionManager3D(BASE_LAYOUT, n_levels=N_LEVELS)
    tile_manager = TileManager(CACHE_INFO)
    coarse_key = BlockKey3D(level=2, gz=0, gy=0, gx=0)
    fine_key = BlockKey3D(level=1, gz=0, gy=0, gx=0)
    tile_manager.stage({coarse_key: 2, fine_key: 1}, frame_number=1)

    lut.rebuild(tile_manager)

    assert lut.lut_data[0, 0, 0, 3] == 1
    assert lut.lut_data[0, 0, 1, 3] == 2
    assert lut.lut_data[2, 0, 0, 3] == 0


def test_fine_brick_slot_coordinates_are_correct() -> None:
    """The fine brick's cell carries its own slot's (sx, sy, sz), not the
    coarse slot's."""
    lut = LutIndirectionManager3D(BASE_LAYOUT, n_levels=N_LEVELS)
    tile_manager = TileManager(CACHE_INFO)
    coarse_key = BlockKey3D(level=2, gz=0, gy=0, gx=0)
    fine_key = BlockKey3D(level=1, gz=0, gy=0, gx=0)
    fill_plan = tile_manager.stage({coarse_key: 2, fine_key: 1}, frame_number=1)

    fine_slot = next(slot for key, slot in fill_plan if key == fine_key)
    sz, sy, sx = fine_slot.grid_pos

    lut.rebuild(tile_manager)

    assert lut.lut_data[0, 0, 0, 0] == sx
    assert lut.lut_data[0, 0, 0, 1] == sy
    assert lut.lut_data[0, 0, 0, 2] == sz
