"""Paint controller for multiscale (zarr-backed) image data stores.

Architecture (Phase 3 — pyramid rebuild + autosave loop)
---------------------------------------------------------
Each brush application:

1. Reads pre-paint values from the open ``WriteBuffer`` (read-your-writes
   through a private tensorstore transaction).
2. Stages the new values into the same buffer.
3. Marks the dirty level-0 bricks in the :class:`WriteLayer`.
4. Writes the new values into the GPU paint cache + LUT via
   :meth:`CellierController.patch_painted_tiles_2d`.

The shader composites the paint cache over the base sample, so painted
voxels are visible on the next frame.  No reslice or eviction occurs
during a brush step.

On commit: coarser LOD bricks are rebuilt bottom-up within the open
transaction, then the transaction is flushed atomically; the GPU paint
textures are cleared; the visible cache is evicted for dirty bricks;
``reslice_visual`` repopulates the base cache from disk.

On autosave: same as commit but undo history is preserved and the
session continues with a fresh transaction.

On abort: ``WriteBuffer`` discards the transaction; the GPU paint
textures are cleared.  No reslice is needed because the base cache is
still pre-paint.

The controller is N-dim agnostic at the storage layer (``WriteLayer`` /
``WriteBuffer``).  Visible feedback is currently only wired for 2-D
displayed-axis configurations; 3-D paint feedback raises
``NotImplementedError`` at construction.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from cellier.v2.paint._abstract import AbstractPaintController
from cellier.v2.paint._write_buffer import TensorStoreWriteBuffer
from cellier.v2.paint._write_layer import BrickKey, WriteLayer

if TYPE_CHECKING:
    from uuid import UUID

    import tensorstore as ts

    from cellier.v2.controller import CellierController
    from cellier.v2.paint._history import PaintStrokeCommand

_PAINT_DEBUG = True


class MultiscalePaintController(AbstractPaintController):
    """Paint controller for OME-Zarr / multiscale-zarr data stores.

    Stages all writes in a tensorstore transaction (RAM-only) until the
    user commits or aborts the session.  Visible feedback is provided by
    writing directly into the GPU paint cache; the shader composites paint
    over the base sample so painted voxels are visible immediately without
    a slicer round-trip.

    Parameters
    ----------
    cellier_controller :
    visual_id :
    scene_id :
    canvas_id :
    data_store :
        Either ``OMEZarrImageDataStore`` or ``MultiscaleZarrDataStore``.
        Must expose ``_ts_stores: list[ts.TensorStore]`` and
        ``level_shapes`` / ``level_transforms``.
    visual_block_size :
        Tile / brick side length used by the visual's render config.
        Must match the ``MultiscaleImageRenderConfig.block_size`` of the
        rendered visual.
    displayed_axes :
        The two data-array axes currently displayed in 2D for the bound
        canvas, in ``(row_axis, col_axis)`` order.  3D paint feedback is
        a follow-up; passing a 3-tuple raises ``NotImplementedError``.
    brush_value :
    brush_radius_voxels :
    history_depth :
    autosave_interval_s :
        If set, a ``QTimer`` fires every this many seconds to flush staged
        paint to disk, rebuild the pyramid, and reset the GPU paint
        textures.  ``None`` disables autosave.
    downsample_mode :
        ``"decimate"`` (default) — stride-2 pick for label stores.
        ``"mean"`` — per-stride average for intensity images.
    """

    def __init__(
        self,
        cellier_controller: CellierController,
        visual_id: UUID,
        scene_id: UUID,
        canvas_id: UUID,
        data_store: Any,
        visual_block_size: int,
        displayed_axes: tuple[int, ...],
        brush_value: float = 1.0,
        brush_radius_voxels: float = 2.0,
        history_depth: int = 100,
        autosave_interval_s: float | None = None,
        downsample_mode: Literal["decimate", "mean"] = "decimate",
    ) -> None:
        if len(displayed_axes) != 2:
            raise NotImplementedError(
                "MultiscalePaintController currently only supports 2-D "
                f"displayed-axis configurations; got {displayed_axes!r}.  "
                "3-D paint feedback is a Phase 3 follow-up."
            )
        if not data_store._ts_stores:
            raise ValueError(
                "data_store has no open tensorstore handles; ensure "
                "model_post_init has run."
            )

        self._data_store = data_store
        self._data_shape: tuple[int, ...] = tuple(
            int(d) for d in data_store.level_shapes[0]
        )
        self._displayed_axes: tuple[int, int] = tuple(displayed_axes)  # type: ignore[assignment]
        self._downsample_mode: str = downsample_mode
        self._autosave_count: int = 0
        self._autosave_interval_s: float | None = autosave_interval_s
        self._last_autosave_time: datetime | None = None
        self._autosave_timer = None  # typed below after QTimer import

        self._write_buffer: TensorStoreWriteBuffer = TensorStoreWriteBuffer(
            data_store._ts_stores[0]
        )
        self._write_layer = WriteLayer(
            data_store_id=data_store.id, block_size=int(visual_block_size)
        )

        super().__init__(
            cellier_controller=cellier_controller,
            visual_id=visual_id,
            scene_id=scene_id,
            canvas_id=canvas_id,
            data_store_id=data_store.id,
            brush_value=brush_value,
            brush_radius_voxels=brush_radius_voxels,
            history_depth=history_depth,
        )

        if autosave_interval_s is not None and autosave_interval_s > 0:
            from PySide6.QtCore import QTimer

            self._autosave_timer = QTimer()
            self._autosave_timer.setInterval(int(autosave_interval_s * 1000))
            self._autosave_timer.timeout.connect(self._do_autosave)
            self._autosave_timer.start()
            if _PAINT_DEBUG:
                print(
                    f"[PAINT-DBG paint] autosave timer started "
                    f"interval={autosave_interval_s}s"
                )

        if _PAINT_DEBUG:
            print(
                f"[PAINT-DBG paint] MultiscalePaintController init "
                f"visual={visual_id} canvas={canvas_id} "
                f"data_shape={self._data_shape} "
                f"block_size={self._write_layer.block_size} "
                f"displayed_axes={self._displayed_axes} "
                f"downsample_mode={downsample_mode}"
            )

    # ------------------------------------------------------------------
    # AbstractPaintController hooks
    # ------------------------------------------------------------------

    def _apply_brush(self, world_coord: np.ndarray) -> None:
        """Re-order world_coord from pygfx (x, y) to data (row, col) order.

        Both image visuals render ``data[r, c]`` at pygfx world ``(x=c, y=r)``.
        ``CellierController._on_raw_pointer_event`` embeds the mouse position
        verbatim: ``world_coord[ax_row] = pygfx_x`` (column index),
        ``world_coord[ax_col] = pygfx_y`` (row index).  Swapping them here
        produces the correct voxel address before the identity
        ``imap_coordinates`` call in the base.
        """
        ax_row, ax_col = self._displayed_axes
        swapped = world_coord.copy()
        swapped[ax_row], swapped[ax_col] = (
            float(world_coord[ax_col]),
            float(world_coord[ax_row]),
        )
        if _PAINT_DEBUG:
            print(
                "[PAINT-DBG paint] _apply_brush swap "
                f"world_in={np.round(world_coord, 2).tolist()} "
                f"swapped={np.round(swapped, 2).tolist()} "
                f"displayed_axes={self._displayed_axes}"
            )
        super()._apply_brush(swapped)

    def _read_old_values(self, voxel_indices: np.ndarray) -> np.ndarray:
        """Read pre-paint values through the open transaction."""
        if voxel_indices.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        old_values = self._write_buffer.read_staged(voxel_indices)
        if _PAINT_DEBUG:
            print(
                f"[PAINT-DBG paint] _read_old_values n={voxel_indices.shape[0]} "
                f"sample={old_values[:5].tolist()}"
            )
        return old_values

    def _write_values(self, voxel_indices: np.ndarray, values: np.ndarray) -> None:
        """Stage writes, mark dirty bricks, patch the GPU paint texture."""
        if voxel_indices.shape[0] == 0:
            return
        self._write_buffer.stage(voxel_indices, values)

        new_dirty = self._write_layer.voxels_to_brick_keys(voxel_indices)
        for key in new_dirty:
            self._write_layer.mark_dirty(key)

        n_patched = self._controller.patch_painted_tiles_2d(
            self._visual_id, voxel_indices, values, self._displayed_axes
        )

        if _PAINT_DEBUG:
            print(
                f"[PAINT-DBG paint] _write_values n={voxel_indices.shape[0]} "
                f"new_dirty_bricks={len(new_dirty)} "
                f"dirty_total={len(self._write_layer.dirty_keys())} "
                f"tiles_patched={n_patched}"
            )

    def _on_stroke_completed(self, command: PaintStrokeCommand) -> None:
        """No background work — visible feedback already happened in _write_values."""

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def commit(self) -> None:
        """Flush staged paint + rebuild pyramid, clear GPU, repopulate cache."""
        if self._autosave_timer is not None:
            self._autosave_timer.stop()

        if _PAINT_DEBUG:
            print(
                f"[PAINT-DBG paint] commit() begin "
                f"dirty_bricks={len(self._write_layer.dirty_keys())} "
                f"history_depth={len(self._history._undo_stack)} "
                f"autosave_count={self._autosave_count}"
            )

        # 1. Rebuild coarser LOD levels within the open transaction.
        rebuild_stats = self._rebuild_pyramid(self._write_buffer.transaction)

        # 2. Flush level-0 + all rebuilt levels atomically.
        self._write_buffer.commit()

        # 3. Drop GPU paint textures — paint is now on disk.
        self._controller.clear_painted_tiles_2d(self._visual_id)

        # 4. Evict + reslice so base cache reflects the committed state.
        self._evict_dirty_visible_tiles()
        self._write_layer.clear()
        self._history._undo_stack.clear()
        self._history._redo_stack.clear()
        self._controller.reslice_visual(self._visual_id)

        if _PAINT_DEBUG:
            print(
                f"[PAINT-DBG paint] commit() complete " f"pyramid_stats={rebuild_stats}"
            )

        self._teardown()

    def abort(self) -> None:
        """Discard staged paint and clear the GPU paint textures."""
        if _PAINT_DEBUG:
            print(
                f"[PAINT-DBG paint] abort() begin "
                f"dirty_bricks={len(self._write_layer.dirty_keys())} "
                f"history_depth={len(self._history._undo_stack)}"
            )
        # 1. Discard staged writes (disk untouched).
        self._write_buffer.abort()

        # 2. Drop the GPU paint textures.
        self._controller.clear_painted_tiles_2d(self._visual_id)

        # 3. NO reslice — the base cache already shows pre-paint data.
        self._write_layer.clear()
        self._history._undo_stack.clear()
        self._history._redo_stack.clear()

        self._teardown()

    def _teardown(self) -> None:
        """Stop autosave timer, unsubscribe bus events, re-enable camera."""
        if self._autosave_timer is not None:
            self._autosave_timer.stop()
            self._autosave_timer = None
            if _PAINT_DEBUG:
                print("[PAINT-DBG paint] autosave timer stopped")
        super()._teardown()

    # ------------------------------------------------------------------
    # Autosave
    # ------------------------------------------------------------------

    def _do_autosave(self) -> None:
        """Flush staged paint to disk, rebuild pyramid, reset GPU textures.

        Called automatically by the autosave timer.  Does NOT clear undo
        history or tear down the session — the session continues with a
        fresh transaction.
        """
        dirty_count = len(self._write_layer.dirty_keys())
        if dirty_count == 0:
            if _PAINT_DEBUG:
                print("[PAINT-DBG paint] _do_autosave() — no dirty bricks, skipping")
            return

        t0 = time.perf_counter()

        if _PAINT_DEBUG:
            print(
                f"[PAINT-DBG paint] _do_autosave() begin "
                f"dirty_bricks={dirty_count} "
                f"autosave_count_before={self._autosave_count}"
            )

        # 1. Rebuild coarser LOD levels within the open transaction.
        rebuild_stats = self._rebuild_pyramid(self._write_buffer.transaction)

        # 2. Flush level-0 + rebuilt levels atomically.
        self._write_buffer.commit()

        # 3. Create a fresh transaction for continued staging.
        self._write_buffer = TensorStoreWriteBuffer(self._data_store._ts_stores[0])

        # 4. Drop GPU paint textures — frees the entire slot pool.
        self._controller.clear_painted_tiles_2d(self._visual_id)

        # 5. Evict base cache for dirty bricks; reslice repopulates from disk.
        self._evict_dirty_visible_tiles()
        self._write_layer.clear()
        self._controller.reslice_visual(self._visual_id)

        self._autosave_count += 1
        self._last_autosave_time = datetime.now()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if _PAINT_DEBUG:
            print(
                f"[PAINT-DBG paint] _do_autosave() complete "
                f"autosave_count={self._autosave_count} "
                f"pyramid_stats={rebuild_stats} "
                f"elapsed={elapsed_ms:.1f}ms"
            )

    # ------------------------------------------------------------------
    # Autosave state accessors
    # ------------------------------------------------------------------

    @property
    def autosave_count(self) -> int:
        """Number of autosaves completed this session."""
        return self._autosave_count

    @property
    def last_autosave_time(self) -> datetime | None:
        """Datetime of the most recent autosave, or None if none yet."""
        return self._last_autosave_time

    # ------------------------------------------------------------------
    # Pyramid rebuild
    # ------------------------------------------------------------------

    def _rebuild_pyramid(self, txn: ts.Transaction | None) -> dict[int, int]:
        """Rebuild coarser LOD bricks for all dirty level-0 bricks.

        Operates within *txn* so the rebuild is atomic with the level-0
        writes.  Reads level k-1 through the transaction (read-your-writes)
        to build level k.

        Parameters
        ----------
        txn :
            The active tensorstore transaction from ``self._write_buffer``.
            Must still be open (before ``commit_sync``).

        Returns
        -------
        dict[int, int]
            Mapping ``level → n_bricks_rebuilt`` for each level k >= 1.
            Empty if the store has only one level or txn is None.
        """
        if txn is None:
            return {}

        n_levels = self._data_store.n_levels
        if n_levels < 2:
            if _PAINT_DEBUG:
                print("[PAINT-DBG pyramid] n_levels=1 — nothing to rebuild")
            return {}

        level_shapes = self._data_store.level_shapes
        level_transforms = self._data_store.level_transforms
        stores = self._data_store._ts_stores
        ax_y, ax_x = self._displayed_axes

        stats: dict[int, int] = {}

        dirty_by_level: dict[int, set[tuple[int, ...]]] = {
            0: {key.grid_coords for key in self._write_layer.dirty_keys()}
        }

        if _PAINT_DEBUG:
            print(
                f"[PAINT-DBG pyramid] rebuild start "
                f"n_levels={n_levels} "
                f"dirty_bricks_level0={len(dirty_by_level[0])}"
            )

        t0 = time.perf_counter()

        for k in range(1, n_levels):
            stride_y = max(
                1,
                round(
                    level_transforms[k].matrix[ax_y, ax_y]
                    / level_transforms[k - 1].matrix[ax_y, ax_y]
                ),
            )
            stride_x = max(
                1,
                round(
                    level_transforms[k].matrix[ax_x, ax_x]
                    / level_transforms[k - 1].matrix[ax_x, ax_x]
                ),
            )

            parent_dirty = dirty_by_level.get(k - 1, set())
            dirty_k: set[tuple[int, ...]] = set()
            for gc in parent_dirty:
                dirty_k.add(
                    tuple(
                        c // stride_y
                        if i == ax_y
                        else c // stride_x
                        if i == ax_x
                        else c
                        for i, c in enumerate(gc)
                    )
                )
            dirty_by_level[k] = dirty_k

            if not dirty_k:
                stats[k] = 0
                continue

            bs = self._write_layer.block_size
            src_store = stores[k - 1].with_transaction(txn)
            dst_store = stores[k].with_transaction(txn)

            shape_km1 = level_shapes[k - 1]
            shape_k = level_shapes[k]

            if _PAINT_DEBUG:
                print(
                    f"[PAINT-DBG pyramid] level {k}: "
                    f"stride=({stride_y},{stride_x}) "
                    f"n_bricks={len(dirty_k)} "
                    f"shape_src={shape_km1} shape_dst={shape_k}"
                )

            n_rebuilt = 0
            for gc in dirty_k:
                gy_k, gx_k = gc[ax_y], gc[ax_x]

                src_y0 = gy_k * bs * stride_y
                src_y1 = min((gy_k + 1) * bs * stride_y, shape_km1[ax_y])
                src_x0 = gx_k * bs * stride_x
                src_x1 = min((gx_k + 1) * bs * stride_x, shape_km1[ax_x])

                if src_y0 >= shape_km1[ax_y] or src_x0 >= shape_km1[ax_x]:
                    continue

                dst_y0 = gy_k * bs
                dst_y1 = min((gy_k + 1) * bs, shape_k[ax_y])
                dst_x0 = gx_k * bs
                dst_x1 = min((gx_k + 1) * bs, shape_k[ax_x])

                if dst_y0 >= shape_k[ax_y] or dst_x0 >= shape_k[ax_x]:
                    continue

                src_slices = [slice(None)] * len(shape_km1)
                src_slices[ax_y] = slice(src_y0, src_y1)
                src_slices[ax_x] = slice(src_x0, src_x1)
                block = np.array(
                    src_store[tuple(src_slices)].read().result(),
                    dtype=np.float32,
                )

                if self._downsample_mode == "decimate":
                    dec_slices = [slice(None)] * block.ndim
                    dec_slices[ax_y] = slice(None, None, stride_y)
                    dec_slices[ax_x] = slice(None, None, stride_x)
                    downsampled = block[tuple(dec_slices)]
                else:  # "mean"
                    h, w = block.shape[ax_y], block.shape[ax_x]
                    h_trim = (h // stride_y) * stride_y
                    w_trim = (w // stride_x) * stride_x
                    trim_slices = [slice(None)] * block.ndim
                    trim_slices[ax_y] = slice(0, h_trim)
                    trim_slices[ax_x] = slice(0, w_trim)
                    block = block[tuple(trim_slices)]
                    downsampled = (
                        block.reshape(
                            h_trim // stride_y,
                            stride_y,
                            w_trim // stride_x,
                            stride_x,
                        )
                        .mean(axis=(1, 3))
                        .astype(np.float32)
                    )

                actual_h = dst_y1 - dst_y0
                actual_w = dst_x1 - dst_x0
                clip_slices = [slice(None)] * downsampled.ndim
                clip_slices[ax_y] = slice(0, actual_h)
                clip_slices[ax_x] = slice(0, actual_w)
                downsampled = downsampled[tuple(clip_slices)]

                dst_slices = [slice(None)] * len(shape_k)
                dst_slices[ax_y] = slice(dst_y0, dst_y1)
                dst_slices[ax_x] = slice(dst_x0, dst_x1)
                dst_store[tuple(dst_slices)].write(downsampled).result()
                n_rebuilt += 1

            stats[k] = n_rebuilt
            if _PAINT_DEBUG:
                print(f"[PAINT-DBG pyramid] level {k}: rebuilt {n_rebuilt} bricks")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if _PAINT_DEBUG:
            print(
                f"[PAINT-DBG pyramid] rebuild complete "
                f"stats={stats} elapsed={elapsed_ms:.1f}ms"
            )
        return stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evict_dirty_visible_tiles(self) -> int:
        """Drop visible cache tiles for every brick currently marked dirty."""
        ax_y, ax_x = self._displayed_axes
        dirty_keys = self._write_layer.dirty_keys()
        dirty_grid_coords_2d: set[tuple[int, int]] = {
            (key.grid_coords[ax_y], key.grid_coords[ax_x]) for key in dirty_keys
        }
        return self._controller.invalidate_painted_tiles_2d(
            self._visual_id, dirty_grid_coords_2d
        )

    def __repr__(self) -> str:
        return (
            f"<MultiscalePaintController visual={self._visual_id} "
            f"canvas={self._canvas_id} "
            f"dirty_bricks={len(self._write_layer.dirty_keys())} "
            f"history_depth={len(self._history._undo_stack)}>"
        )


# Re-export BrickKey so callers can reach it via the paint package without
# touching private modules.
__all__ = ["BrickKey", "MultiscalePaintController"]
