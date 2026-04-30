"""Integration tests for TensorStoreWriteBuffer.

These exercise tensorstore Transaction read-your-writes semantics
against a temporary on-disk zarr.  No Qt event loop is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tensorstore as ts

from cellier.v2.paint import TensorStoreWriteBuffer

if TYPE_CHECKING:
    import pathlib


@pytest.fixture
def tmp_zarr(tmp_path: pathlib.Path) -> ts.TensorStore:
    """A fresh, empty 16x16 float32 zarr store on disk."""
    store_path = tmp_path / "test.zarr"
    store = ts.open(
        {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(store_path)},
            "metadata": {
                "shape": [16, 16],
                "chunks": [8, 8],
                "dtype": "<f4",
            },
            "create": True,
        }
    ).result()
    return store


def test_stage_then_read_staged_returns_written_values(
    tmp_zarr: ts.TensorStore,
) -> None:
    buf = TensorStoreWriteBuffer(tmp_zarr)
    voxels = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    new_vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # Pre-paint values are zero (fresh store).
    pre = buf.read_staged(voxels)
    np.testing.assert_array_equal(pre, np.zeros(3, dtype=np.float32))

    buf.stage(voxels, new_vals)
    post = buf.read_staged(voxels)
    np.testing.assert_array_equal(post, new_vals)


def test_staged_writes_invisible_outside_transaction(
    tmp_zarr: ts.TensorStore,
) -> None:
    buf = TensorStoreWriteBuffer(tmp_zarr)
    voxels = np.array([[1, 2], [3, 4]], dtype=np.int64)
    buf.stage(voxels, np.array([7.0, 8.0], dtype=np.float32))

    # Reading the underlying store directly (no transaction) must not see the
    # staged writes.
    raw = tmp_zarr.vindex[(voxels[:, 0], voxels[:, 1])].read().result()
    np.testing.assert_array_equal(np.asarray(raw), np.zeros(2, dtype=np.float32))


def test_commit_persists_writes(tmp_zarr: ts.TensorStore) -> None:
    buf = TensorStoreWriteBuffer(tmp_zarr)
    voxels = np.array([[1, 2], [5, 6]], dtype=np.int64)
    new_vals = np.array([4.0, 5.0], dtype=np.float32)
    buf.stage(voxels, new_vals)
    buf.commit()

    # After commit the underlying store reflects the writes.
    raw = tmp_zarr.vindex[(voxels[:, 0], voxels[:, 1])].read().result()
    np.testing.assert_array_equal(np.asarray(raw), new_vals)


def test_abort_discards_writes(tmp_zarr: ts.TensorStore) -> None:
    buf = TensorStoreWriteBuffer(tmp_zarr)
    voxels = np.array([[1, 2]], dtype=np.int64)
    buf.stage(voxels, np.array([99.0], dtype=np.float32))
    buf.abort()

    raw = tmp_zarr.vindex[(voxels[:, 0], voxels[:, 1])].read().result()
    np.testing.assert_array_equal(np.asarray(raw), np.zeros(1, dtype=np.float32))


def test_stage_after_commit_raises(tmp_zarr: ts.TensorStore) -> None:
    buf = TensorStoreWriteBuffer(tmp_zarr)
    buf.commit()
    with pytest.raises(RuntimeError):
        buf.stage(
            np.array([[0, 0]], dtype=np.int64),
            np.array([1.0], dtype=np.float32),
        )


def test_stage_after_abort_raises(tmp_zarr: ts.TensorStore) -> None:
    buf = TensorStoreWriteBuffer(tmp_zarr)
    buf.abort()
    with pytest.raises(RuntimeError):
        buf.stage(
            np.array([[0, 0]], dtype=np.int64),
            np.array([1.0], dtype=np.float32),
        )


def test_double_commit_is_noop(tmp_zarr: ts.TensorStore) -> None:
    buf = TensorStoreWriteBuffer(tmp_zarr)
    buf.commit()
    buf.commit()  # must not raise


def test_double_abort_is_noop(tmp_zarr: ts.TensorStore) -> None:
    buf = TensorStoreWriteBuffer(tmp_zarr)
    buf.abort()
    buf.abort()  # must not raise


def test_empty_stage_is_noop(tmp_zarr: ts.TensorStore) -> None:
    buf = TensorStoreWriteBuffer(tmp_zarr)
    buf.stage(np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32))
    # No exception, and a subsequent read returns zeros.
    voxels = np.array([[0, 0]], dtype=np.int64)
    assert buf.read_staged(voxels)[0] == 0.0


def test_overwrite_same_voxel_keeps_latest(tmp_zarr: ts.TensorStore) -> None:
    buf = TensorStoreWriteBuffer(tmp_zarr)
    voxels = np.array([[3, 3]], dtype=np.int64)
    buf.stage(voxels, np.array([1.0], dtype=np.float32))
    buf.stage(voxels, np.array([2.0], dtype=np.float32))
    assert buf.read_staged(voxels)[0] == 2.0
