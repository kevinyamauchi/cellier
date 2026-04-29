"""Paint controller for multiscale (zarr-backed) image data stores.

Architecture (Phase 2 — GPU paint-texture fast path)
----------------------------------------------------
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

On commit: ``WriteBuffer`` flushes the transaction to disk; the GPU
paint textures are cleared; the visible cache is evicted for dirty
bricks; ``reslice_visual`` repopulates the base cache from disk so the
post-session display reflects the now-persisted state.

On abort: ``WriteBuffer`` discards the transaction; the GPU paint
textures are cleared.  No reslice is needed because the base cache is
still pre-paint.

The controller is N-dim agnostic at the storage layer (``WriteLayer`` /
``WriteBuffer``).  Visible feedback is currently only wired for 2-D
displayed-axis configurations; 3-D paint feedback raises
``NotImplementedError`` at construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from cellier.v2.paint._abstract import AbstractPaintController
from cellier.v2.paint._write_buffer import TensorStoreWriteBuffer
from cellier.v2.paint._write_layer import BrickKey, WriteLayer

if TYPE_CHECKING:
    from uuid import UUID

    from cellier.v2.controller import CellierController
    from cellier.v2.paint._history import PaintStrokeCommand

_PAINT_DEBUG = True


class MultiscalePaintController(AbstractPaintController):
    """Paint controller for OME-Zarr / multiscale-zarr data stores.

    Stages all writes in a tensorstore transaction (RAM-only) until the
    user commits or aborts the session.  Visible feedback is provided by
    evicting the affected tiles from the GPU tile cache and triggering
    a visual reslice — the slicer re-fetches the evicted tiles through
    the open transaction so painted voxels are visible immediately
    after one async round-trip.

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
        rendered visual; the paint controller uses it to map level-0
        voxels to brick keys for both the write layer and the cache
        eviction step.
    displayed_axes :
        The two data-array axes currently displayed in 2D for the bound
        canvas, in ``(row_axis, col_axis)`` order.  3D paint feedback is
        a follow-up; passing a 3-tuple raises ``NotImplementedError``.
    brush_value :
    brush_radius_voxels :
    history_depth :
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
        # Level-0 ndim and shape — used by AbstractPaintController._voxels_in_radius
        # and by the brick-key map.
        self._data_shape: tuple[int, ...] = tuple(
            int(d) for d in data_store.level_shapes[0]
        )
        self._displayed_axes: tuple[int, int] = tuple(displayed_axes)  # type: ignore[assignment]

        # The buffer's internal transaction is private to it — the slicer
        # reads from the unwrapped data store and never sees staged paint.
        # Visible feedback comes from the GPU paint texture instead.
        self._write_buffer: TensorStoreWriteBuffer = TensorStoreWriteBuffer(
            data_store._ts_stores[0]
        )

        # Sparse dirty-brick tracker.
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

        if _PAINT_DEBUG:
            print(
                f"[PAINT-DBG paint] MultiscalePaintController init "
                f"visual={visual_id} canvas={canvas_id} "
                f"data_shape={self._data_shape} "
                f"block_size={self._write_layer.block_size} "
                f"displayed_axes={self._displayed_axes}"
            )

    # ------------------------------------------------------------------
    # AbstractPaintController hooks
    # ------------------------------------------------------------------

    def _apply_brush(self, world_coord: np.ndarray) -> None:
        """Re-order world_coord for the multiscale 2-D upload convention.

        ``GFXImageMemoryVisual.on_data_ready_2d`` transposes the array on
        upload, so its convention is "data axis 0 ↔ pygfx-x"; the
        coordinate embedding in
        ``CellierController._on_raw_pointer_event`` matches that.

        ``GFXMultiscaleImageVisual._build_2d_node`` does **not** transpose:
        the 2-D ``gfx.Image`` is sized with ``local.scale = (W, H, 1)``
        and the LUT shader maps ``texcoord.x → vol_size_x = W``.  Net
        effect: ``data[r, c]`` is rendered at pygfx world ``(x=c, y=r)``
        — i.e. data axis 0 ↔ pygfx-y, the opposite of the in-memory
        visual.  Without the swap below, painting at upper-left would
        place paint at lower-right (a transpose).

        For 2-D the fix is to swap the two displayed-axis components of
        ``world_coord`` once, here, before delegating to the shared
        brush logic.  The voxel-radius computation, dirty-brick mapping,
        and tile invalidation all run in voxel space after the swap and
        need no further changes.

        3-D paint feedback (a Phase 3 follow-up) needs its own check:
        the multiscale 3-D upload uses the standard pygfx ``(x, y, z)``
        order over data ``(z, y, x)``, so the equivalent fix there is a
        full ``[::-1]`` reversal of the displayed-axis components, not
        just a 2-axis swap.  Guarded against in ``__init__`` for now.
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
        """No background work — visible feedback already happened in _write_values.

        Pyramid rebuild and autosave are deferred to a follow-up.
        """

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def commit(self) -> None:
        """Flush staged paint to disk, clear GPU paint, repopulate base cache.

        Blocks the calling thread until the tensorstore transaction has
        been written to disk.  For typical session sizes (tens of
        kilobytes to a few megabytes) this is fast; large sessions may
        stall the UI for hundreds of milliseconds.  Adding an autosave
        loop is a Phase 3 follow-up.
        """
        if _PAINT_DEBUG:
            print(
                f"[PAINT-DBG paint] commit() begin "
                f"dirty_bricks={len(self._write_layer.dirty_keys())} "
                f"history_depth={len(self._history._undo_stack)}"
            )
        # 1. Flush to disk via the buffer's transaction.
        self._write_buffer.commit()

        # 2. Drop the GPU paint textures — paint is now on disk.
        self._controller.clear_painted_tiles_2d(self._visual_id)

        # 3. Evict the base cache for dirty bricks; reslice repopulates from
        #    disk so the post-session display matches the persisted state.
        self._evict_dirty_visible_tiles()

        self._write_layer.clear()
        self._history._undo_stack.clear()
        self._history._redo_stack.clear()

        self._controller.reslice_visual(self._visual_id)
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
