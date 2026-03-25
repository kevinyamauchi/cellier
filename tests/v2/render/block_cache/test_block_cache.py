"""Tests for BlockCache3D and LutIndirectionManager."""

import numpy as np
import pytest

from cellier.v2.render.block_cache import (
    BlockCache3D,
    BlockKey3D,
    compute_block_cache_parameters_3d,
)

# Cache used throughout: block_size=4, overlap=1, grid_side=2, 8 slots (7 usable).
CACHE_INFO = compute_block_cache_parameters_3d(
    block_size=4, gpu_budget_bytes=8 * 6**3 * 4
)


def _stage_and_commit(cache: BlockCache3D, bricks: dict, frame_number: int):
    """Stage bricks and immediately commit all misses (synchronous helper)."""
    fill_plan = cache.stage(bricks, frame_number=frame_number)
    for key, slot in fill_plan:
        cache.tile_manager.commit(key, slot)
    return fill_plan


def test_first_request_is_a_miss() -> None:
    cache = BlockCache3D(CACHE_INFO)
    key = BlockKey3D(level=1, gz=0, gy=0, gx=0)
    fill_plan = cache.stage({key: 1}, frame_number=1)
    assert len(fill_plan) == 1
    assert fill_plan[0][0] == key


def test_repeated_request_is_a_hit() -> None:
    cache = BlockCache3D(CACHE_INFO)
    key = BlockKey3D(level=1, gz=0, gy=0, gx=0)
    _stage_and_commit(cache, {key: 1}, frame_number=1)
    fill_plan = cache.stage({key: 1}, frame_number=2)
    assert fill_plan == []


def test_hit_does_not_change_slot() -> None:
    cache = BlockCache3D(CACHE_INFO)
    key = BlockKey3D(level=1, gz=0, gy=0, gx=0)
    first_plan = _stage_and_commit(cache, {key: 1}, frame_number=1)
    first_slot = first_plan[0][1]

    cache.stage({key: 1}, frame_number=2)
    same_slot = cache.tile_manager.tilemap[key]

    assert same_slot.index == first_slot.index
    assert same_slot.grid_pos == first_slot.grid_pos


def test_lru_evicts_oldest_brick() -> None:
    """Fill cache with bricks A-G at frames 1-7, refresh B at frame 8,
    then request H. A should be evicted (oldest timestamp)."""
    cache = BlockCache3D(CACHE_INFO)
    keys = [BlockKey3D(level=1, gz=i, gy=0, gx=0) for i in range(7)]
    for frame, key in enumerate(keys, start=1):
        _stage_and_commit(cache, {key: 1}, frame_number=frame)

    key_a = keys[0]
    slot_a_index = cache.tile_manager.tilemap[key_a].index

    key_b = keys[1]
    cache.stage({key_b: 1}, frame_number=8)  # hit — no commit needed

    key_h = BlockKey3D(level=1, gz=99, gy=0, gx=0)
    fill_plan = cache.stage({key_h: 1}, frame_number=9)

    assert len(fill_plan) == 1
    assert fill_plan[0][1].index == slot_a_index
    assert key_a not in cache.tile_manager.tilemap


def test_write_brick_fills_correct_slice() -> None:
    cache = BlockCache3D(CACHE_INFO)
    key = BlockKey3D(level=1, gz=0, gy=0, gx=0)
    fill_plan = cache.stage({key: 1}, frame_number=1)
    slot = fill_plan[0][1]

    pbs = cache.info.padded_block_size
    data = np.full((pbs, pbs, pbs), fill_value=7.0, dtype=np.float32)
    cache.write_brick(slot, data)

    sz, sy, sx = slot.grid_pos
    z0, y0, x0 = sz * pbs, sy * pbs, sx * pbs
    assert np.all(cache.cache_data[z0 : z0 + pbs, y0 : y0 + pbs, x0 : x0 + pbs] == 7.0)


def test_write_brick_does_not_touch_other_slots() -> None:
    cache = BlockCache3D(CACHE_INFO)
    key = BlockKey3D(level=1, gz=0, gy=0, gx=0)
    fill_plan = cache.stage({key: 1}, frame_number=1)
    slot = fill_plan[0][1]

    pbs = cache.info.padded_block_size
    cache.write_brick(slot, np.ones((pbs, pbs, pbs), dtype=np.float32))

    assert cache.cache_data.sum() == pytest.approx(float(pbs**3))
