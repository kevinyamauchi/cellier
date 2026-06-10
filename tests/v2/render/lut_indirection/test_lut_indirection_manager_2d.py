"""Tests for the 2D LUT indirection rebuild, focusing on the two-phase sweep
and viewport clipping of stale (old-slice) background tiles."""

import numpy as np

from cellier.v2.render.block_cache._cache_parameters_2d import (
    compute_block_cache_parameters_2d,
)
from cellier.v2.render.block_cache._tile_manager_2d import BlockKey2D, TileManager2D
from cellier.v2.render.lut_indirection._layout_2d import BlockLayout2D
from cellier.v2.render.lut_indirection._lut_indirection_manager_2d import (
    LutIndirectionManager2D,
)

# 4x4 finest grid (block_size=1), single LOD level so each tile covers one cell.
BASE_LAYOUT = BlockLayout2D.from_shape(shape=(4, 4), block_size=1, overlap=1)
N_LEVELS = 1

CACHE_INFO = compute_block_cache_parameters_2d(
    gpu_budget_bytes=64 * 3 * 3 * 4, block_size=1, overlap=1
)

OLD_SLICE = ((0, 10),)
NEW_SLICE = ((0, 20),)


def _make_manager() -> LutIndirectionManager2D:
    return LutIndirectionManager2D(BASE_LAYOUT, n_levels=N_LEVELS)


def _stage_and_commit(tile_manager: TileManager2D, required: dict, frame: int):
    fill_plan = tile_manager.stage(required, frame_number=frame)
    for key, slot in fill_plan:
        tile_manager.commit(key, slot)
    return fill_plan


def test_background_tile_in_view_is_written() -> None:
    """An old-slice tile inside the viewport is kept as a placeholder."""
    lut = _make_manager()
    tm = TileManager2D(CACHE_INFO)
    bg_key = BlockKey2D(level=1, g0=0, g1=0, slice_coord=OLD_SLICE)
    _stage_and_commit(tm, {bg_key: 1}, frame=1)

    lut.rebuild(
        tm,
        current_slice_coord=NEW_SLICE,
        viewport_cells=(0, 0, 2, 2),  # cells [0:2, 0:2] visible
    )

    assert lut.lut_data[0, 0, 2] == 1  # level channel: referenced


def test_background_tile_out_of_view_is_clipped() -> None:
    """An old-slice tile outside the viewport is no longer referenced."""
    lut = _make_manager()
    tm = TileManager2D(CACHE_INFO)
    bg_key = BlockKey2D(level=1, g0=3, g1=3, slice_coord=OLD_SLICE)
    _stage_and_commit(tm, {bg_key: 1}, frame=1)

    lut.rebuild(
        tm,
        current_slice_coord=NEW_SLICE,
        viewport_cells=(0, 0, 2, 2),  # cell (3, 3) is outside
    )

    # The tile stays resident in the cache (not evicted) ...
    assert bg_key in tm.tilemap
    # ... but is not referenced anywhere in the LUT.
    assert np.all(lut.lut_data[..., 2] == 0)


def test_foreground_tile_out_of_view_is_not_clipped() -> None:
    """Current-slice (foreground) tiles are written regardless of viewport."""
    lut = _make_manager()
    tm = TileManager2D(CACHE_INFO)
    fg_key = BlockKey2D(level=1, g0=3, g1=3, slice_coord=NEW_SLICE)
    _stage_and_commit(tm, {fg_key: 1}, frame=1)

    lut.rebuild(
        tm,
        current_slice_coord=NEW_SLICE,
        viewport_cells=(0, 0, 2, 2),  # cell (3, 3) outside, but FG is not clipped
    )

    assert lut.lut_data[3, 3, 2] == 1


def test_no_viewport_keeps_legacy_background_behavior() -> None:
    """viewport_cells=None writes background tiles across the full grid."""
    lut = _make_manager()
    tm = TileManager2D(CACHE_INFO)
    bg_key = BlockKey2D(level=1, g0=3, g1=3, slice_coord=OLD_SLICE)
    _stage_and_commit(tm, {bg_key: 1}, frame=1)

    lut.rebuild(tm, current_slice_coord=NEW_SLICE, viewport_cells=None)

    assert lut.lut_data[3, 3, 2] == 1


def test_foreground_overwrites_clipped_background_in_view() -> None:
    """Where both exist in view, the current-slice tile wins."""
    lut = _make_manager()
    tm = TileManager2D(CACHE_INFO)
    bg_key = BlockKey2D(level=1, g0=0, g1=0, slice_coord=OLD_SLICE)
    fg_key = BlockKey2D(level=1, g0=0, g1=0, slice_coord=NEW_SLICE)
    fill = _stage_and_commit(tm, {bg_key: 1, fg_key: 1}, frame=1)
    fg_slot = next(slot for key, slot in fill if key == fg_key)
    sy, sx = fg_slot.grid_pos

    lut.rebuild(tm, current_slice_coord=NEW_SLICE, viewport_cells=(0, 0, 2, 2))

    assert lut.lut_data[0, 0, 0] == sx
    assert lut.lut_data[0, 0, 1] == sy
    assert lut.lut_data[0, 0, 2] == 1
