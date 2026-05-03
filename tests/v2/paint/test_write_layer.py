"""Unit tests for WriteLayer / BrickKey."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier.v2.paint import BrickKey, WriteLayer


def test_voxel_to_brick_key_2d() -> None:
    layer = WriteLayer(data_store_id=uuid4(), block_size=32)
    assert layer.voxel_to_brick_key(np.array([0, 0])) == BrickKey(
        level=0, grid_coords=(0, 0)
    )
    assert layer.voxel_to_brick_key(np.array([31, 31])) == BrickKey(
        level=0, grid_coords=(0, 0)
    )
    assert layer.voxel_to_brick_key(np.array([32, 31])) == BrickKey(
        level=0, grid_coords=(1, 0)
    )
    assert layer.voxel_to_brick_key(np.array([0, 32])) == BrickKey(
        level=0, grid_coords=(0, 1)
    )
    assert layer.voxel_to_brick_key(np.array([100, 200])) == BrickKey(
        level=0, grid_coords=(3, 6)
    )


def test_voxel_to_brick_key_3d() -> None:
    layer = WriteLayer(data_store_id=uuid4(), block_size=16)
    assert layer.voxel_to_brick_key(np.array([0, 0, 0])) == BrickKey(
        level=0, grid_coords=(0, 0, 0)
    )
    assert layer.voxel_to_brick_key(np.array([16, 0, 0])) == BrickKey(
        level=0, grid_coords=(1, 0, 0)
    )
    assert layer.voxel_to_brick_key(np.array([5, 17, 47])) == BrickKey(
        level=0, grid_coords=(0, 1, 2)
    )


def test_voxels_to_brick_keys_unique() -> None:
    layer = WriteLayer(data_store_id=uuid4(), block_size=8)
    voxels = np.array([[0, 0], [1, 1], [7, 7], [8, 8], [9, 9]], dtype=np.int64)
    keys = layer.voxels_to_brick_keys(voxels)
    assert keys == {
        BrickKey(level=0, grid_coords=(0, 0)),
        BrickKey(level=0, grid_coords=(1, 1)),
    }


def test_voxels_to_brick_keys_empty() -> None:
    layer = WriteLayer(data_store_id=uuid4(), block_size=8)
    assert layer.voxels_to_brick_keys(np.zeros((0, 2), dtype=np.int64)) == set()


def test_dirty_tracking_lifecycle() -> None:
    layer = WriteLayer(data_store_id=uuid4(), block_size=4)
    k1 = BrickKey(level=0, grid_coords=(0, 0))
    k2 = BrickKey(level=0, grid_coords=(1, 0))
    assert not layer.is_dirty(k1)
    assert layer.dirty_keys() == set()

    layer.mark_dirty(k1)
    assert layer.is_dirty(k1)
    assert not layer.is_dirty(k2)

    # Repeated mark_dirty is idempotent.
    layer.mark_dirty(k1)
    layer.mark_dirty(k2)
    assert layer.dirty_keys() == {k1, k2}

    # dirty_keys() returns a copy — caller can mutate without affecting state.
    snapshot = layer.dirty_keys()
    snapshot.clear()
    assert layer.dirty_keys() == {k1, k2}

    layer.clear()
    assert layer.dirty_keys() == set()
    assert not layer.is_dirty(k1)


def test_invalid_block_size() -> None:
    with pytest.raises(ValueError):
        WriteLayer(data_store_id=uuid4(), block_size=0)
    with pytest.raises(ValueError):
        WriteLayer(data_store_id=uuid4(), block_size=-1)
