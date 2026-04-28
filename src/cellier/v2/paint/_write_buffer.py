"""Staged write buffer for the multiscale paint controller.

The :class:`WriteBuffer` protocol lets paint controllers stage voxel
writes in memory and either flush them to durable storage (``commit``)
or discard them (``abort``).  The default :class:`TensorStoreWriteBuffer`
uses a :class:`tensorstore.Transaction` to provide read-your-writes
semantics: any voxel staged via ``stage`` is observable via
``read_staged`` (and via any read on the wrapped store) until the
transaction is committed or aborted.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import tensorstore as ts

_PAINT_DEBUG = True


def _voxel_indices_to_vindex(
    voxel_indices: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Convert an ``(N, ndim)`` int array to a tuple of per-axis index arrays.

    This is the form accepted by ``tensorstore.TensorStore.vindex`` and by
    ``numpy``'s advanced fancy indexing (``arr[ix0, ix1, ...]``).
    """
    return tuple(
        voxel_indices[:, i].astype(np.int64) for i in range(voxel_indices.shape[1])
    )


@runtime_checkable
class WriteBuffer(Protocol):
    """Protocol for staged voxel writes against a level-0 store.

    Concrete implementations stage writes in RAM, allow read-your-writes
    visibility, and flush or discard atomically.

    Implementations must be safe to call from the Qt main thread; the
    paint controller never invokes them from a background thread.
    """

    def stage(self, voxel_indices: np.ndarray, values: np.ndarray) -> None:
        """Stage a vector of point writes.

        Parameters
        ----------
        voxel_indices :
            Shape ``(N, ndim)`` int64.  Each row is one level-0 voxel
            index in data-array axis order.
        values :
            Shape ``(N,)``.  Values to write.  Caller is responsible for
            providing a dtype the underlying store accepts (typically
            float32 cast on write).
        """

    def read_staged(self, voxel_indices: np.ndarray) -> np.ndarray:
        """Return the *current* values at the given voxel indices.

        With read-your-writes semantics, this returns staged values for
        voxels that have been written through ``stage``, falling back to
        the underlying store for unwritten voxels.

        Returns shape ``(N,)`` float32.
        """

    def commit(self) -> None:
        """Flush all staged writes to durable storage.  Blocks until done."""

    def abort(self) -> None:
        """Discard all staged writes.  Returns the buffer to a fresh state."""


class TensorStoreWriteBuffer:
    """:class:`WriteBuffer` backed by a :class:`tensorstore.Transaction`.

    Wraps a single TensorStore handle (typically the level-0 store of a
    multiscale data store) in an isolated transaction.  All writes are
    coalesced in tensorstore's in-memory write buffer until ``commit``
    or ``abort`` is called.

    Parameters
    ----------
    store :
        The tensorstore handle to paint against.  This must be the
        level-0 store of the multiscale data store.

    Notes
    -----
    The transaction is created in *isolated* mode (the default) which
    is what we want: paints are not visible outside the transaction
    until ``commit``.

    After ``commit`` or ``abort`` the buffer is one-shot — calling
    ``stage`` again will fail since the transaction has been finalised.
    The paint controller wraps lifecycle management around this.
    """

    def __init__(
        self,
        store: ts.TensorStore,
        transaction: ts.Transaction | None = None,
    ) -> None:
        """Construct.

        Parameters
        ----------
        store :
            Untransacted level-0 store handle.
        transaction :
            Optional pre-built transaction.  When provided, the same
            transaction can be shared with other consumers (e.g. the
            data store's read path) so they observe staged writes.
            When ``None``, a fresh transaction is created.
        """
        self._store_no_txn = store
        if transaction is None:
            transaction = ts.Transaction()
        self._txn: ts.Transaction | None = transaction
        self._store: ts.TensorStore | None = store.with_transaction(transaction)
        # Track per-voxel stage count so debug output is meaningful.
        self._n_staged_writes: int = 0
        if _PAINT_DEBUG:
            print(
                "[PAINT-DBG paint] TensorStoreWriteBuffer opened "
                f"shape={tuple(store.domain.shape)} dtype={store.dtype}"
            )

    @property
    def transaction(self) -> ts.Transaction | None:
        """The active transaction, or None after commit/abort."""
        return self._txn

    @property
    def transactional_store(self) -> ts.TensorStore | None:
        """The store wrapped in this buffer's transaction (read-your-writes)."""
        return self._store

    # ------------------------------------------------------------------
    # WriteBuffer protocol
    # ------------------------------------------------------------------

    def stage(self, voxel_indices: np.ndarray, values: np.ndarray) -> None:
        if self._store is None:
            raise RuntimeError(
                "TensorStoreWriteBuffer.stage() called after commit/abort."
            )
        if voxel_indices.shape[0] == 0:
            return
        idx = _voxel_indices_to_vindex(voxel_indices)
        # Cast to the underlying store dtype; tensorstore raises if the
        # dtype is incompatible.
        target_dtype = self._store.dtype.numpy_dtype
        cast_values = np.asarray(values).astype(target_dtype, copy=False)
        # vindex.write() returns a future; .result() blocks until the
        # in-memory write buffer has accepted the data.
        self._store.vindex[idx].write(cast_values).result()
        self._n_staged_writes += int(voxel_indices.shape[0])
        if _PAINT_DEBUG:
            print(
                f"[PAINT-DBG paint] write_buffer.stage n={voxel_indices.shape[0]} "
                f"running_total={self._n_staged_writes}"
            )

    def read_staged(self, voxel_indices: np.ndarray) -> np.ndarray:
        if self._store is None:
            raise RuntimeError(
                "TensorStoreWriteBuffer.read_staged() called after commit/abort."
            )
        if voxel_indices.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        idx = _voxel_indices_to_vindex(voxel_indices)
        # vindex read through the transaction sees both staged writes
        # and underlying data.
        raw = self._store.vindex[idx].read().result()
        return np.asarray(raw, dtype=np.float32)

    def commit(self) -> None:
        if self._txn is None:
            return
        if _PAINT_DEBUG:
            print(
                f"[PAINT-DBG paint] write_buffer.commit "
                f"staged_voxels={self._n_staged_writes}"
            )
        self._txn.commit_sync()
        self._txn = None
        self._store = None

    def abort(self) -> None:
        if self._txn is None:
            return
        if _PAINT_DEBUG:
            print(
                f"[PAINT-DBG paint] write_buffer.abort "
                f"staged_voxels={self._n_staged_writes}"
            )
        self._txn.abort()
        self._txn = None
        self._store = None
