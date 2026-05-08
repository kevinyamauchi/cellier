"""Tests for int32-dtype BlockCache2D and BlockCache3D for label rendering."""

from __future__ import annotations

import numpy as np

from cellier.v2.render.block_cache import (
    BlockCache3D,
    compute_block_cache_parameters_3d,
)
from cellier.v2.render.block_cache._block_cache_2d import BlockCache2D
from cellier.v2.render.block_cache._cache_parameters_2d import (
    build_cache_texture_2d,
    compute_block_cache_parameters_2d,
)

# ── 2D int32 cache ────────────────────────────────────────────────────────────


def test_build_cache_texture_2d_int32():
    params = compute_block_cache_parameters_2d(
        gpu_budget_bytes=4 * 1024**2,
        block_size=16,
        overlap=1,
    )
    data, tex = build_cache_texture_2d(params, dtype=np.int32)
    assert data.dtype == np.int32
    # pygfx Texture object should be created without error
    assert tex is not None


def test_build_cache_texture_2d_float32_default():
    params = compute_block_cache_parameters_2d(
        gpu_budget_bytes=4 * 1024**2,
        block_size=16,
    )
    data, tex = build_cache_texture_2d(params)
    assert data.dtype == np.float32


def test_block_cache_2d_int32_write():
    params = compute_block_cache_parameters_2d(
        gpu_budget_bytes=4 * 1024**2,
        block_size=16,
        overlap=1,
    )
    cache = BlockCache2D(cache_parameters=params, dtype=np.int32)
    assert cache.cache_data.dtype == np.int32

    # Stage a slot and write
    from cellier.v2.render.block_cache._tile_manager_2d import BlockKey2D

    key = BlockKey2D(level=1, g0=0, g1=0, slice_coord=())
    fill_plan = cache.tile_manager.stage({key: None}, frame_number=1)
    assert len(fill_plan) == 1
    tile_key, slot = fill_plan[0]

    pbs = params.padded_block_size
    data = np.arange(pbs * pbs, dtype=np.int32).reshape(pbs, pbs)
    cache.write_tile(slot, data, key=tile_key)

    sy, sx = slot.grid_pos
    y0 = sy * pbs
    x0 = sx * pbs
    np.testing.assert_array_equal(cache.cache_data[y0 : y0 + pbs, x0 : x0 + pbs], data)


# ── 3D int32 cache ────────────────────────────────────────────────────────────


def test_block_cache_3d_int32():
    params = compute_block_cache_parameters_3d(
        block_size=8,
        gpu_budget_bytes=4 * 1024**2,
        overlap=1,
        dtype=np.int32,
    )
    cache = BlockCache3D(cache_parameters=params, dtype=np.int32)
    assert cache.cache_data.dtype == np.int32


def test_block_cache_3d_int32_write_brick():
    params = compute_block_cache_parameters_3d(
        block_size=8,
        gpu_budget_bytes=4 * 1024**2,
        overlap=1,
        dtype=np.int32,
    )
    cache = BlockCache3D(cache_parameters=params, dtype=np.int32)

    from cellier.v2.render.block_cache import BlockKey3D

    key = BlockKey3D(level=1, g0=0, g1=0, g2=0)
    fill_plan = cache.tile_manager.stage({key: None}, frame_number=1)
    _, slot = fill_plan[0]

    pbs = params.padded_block_size
    data = np.zeros((pbs, pbs, pbs), dtype=np.int32)
    data[2, 2, 2] = 42

    # write_brick with background_label=0 → contains non-background
    contains = cache.write_brick(slot, data, key=key, background_label=0)
    assert contains is True


def test_block_cache_3d_int32_all_background():
    params = compute_block_cache_parameters_3d(
        block_size=8,
        gpu_budget_bytes=4 * 1024**2,
        overlap=1,
        dtype=np.int32,
    )
    cache = BlockCache3D(cache_parameters=params, dtype=np.int32)

    from cellier.v2.render.block_cache import BlockKey3D

    key = BlockKey3D(level=1, g0=0, g1=0, g2=0)
    fill_plan = cache.tile_manager.stage({key: None}, frame_number=1)
    _, slot = fill_plan[0]

    pbs = params.padded_block_size
    data = np.zeros((pbs, pbs, pbs), dtype=np.int32)  # all background

    contains = cache.write_brick(slot, data, key=key, background_label=0)
    assert contains is False


def test_block_cache_3d_float32_default_unchanged():
    """Ensure float32 default is preserved after adding int32 support."""
    params = compute_block_cache_parameters_3d(
        block_size=8,
        gpu_budget_bytes=4 * 1024**2,
        overlap=3,
    )
    cache = BlockCache3D(cache_parameters=params)
    assert cache.cache_data.dtype == np.float32
