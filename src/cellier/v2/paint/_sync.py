"""Synchronous in-memory paint controller for ImageMemoryStore and LabelMemoryStore."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cellier.v2.paint._abstract import AbstractPaintController
from cellier.v2.paint._history import ActiveStroke

if TYPE_CHECKING:
    from uuid import UUID

    from cellier.v2.controller import CellierController
    from cellier.v2.data.image._image_memory_store import ImageMemoryStore
    from cellier.v2.data.label._label_memory_store import LabelMemoryStore
    from cellier.v2.events._events import CanvasMousePress2DEvent
    from cellier.v2.paint._history import PaintStrokeCommand


class SyncPaintController(AbstractPaintController):
    """Paint controller for :class:`ImageMemoryStore` and :class:`LabelMemoryStore`.

    Directly and synchronously mutates the backing numpy array on every
    brush application, then calls
    :meth:`CellierController.reslice_scene` so the display updates
    immediately.  Suitable for testing and in-memory annotation
    workflows.

    Parameters
    ----------
    cellier_controller : CellierController
    visual_id : UUID
    scene_id : UUID
    canvas_id : UUID
    data_store : ImageMemoryStore | LabelMemoryStore
        The store whose ``.data`` array is painted directly.
    displayed_axes : tuple[int, int]
        The two data-array axes currently displayed in 2D for the bound
        canvas, in ``(row_axis, col_axis)`` order.
    brush_value : float
    brush_radius_voxels : float
    history_depth : int
    """

    def __init__(
        self,
        cellier_controller: CellierController,
        visual_id: UUID,
        scene_id: UUID,
        canvas_id: UUID,
        data_store: ImageMemoryStore | LabelMemoryStore,
        displayed_axes: tuple[int, ...],
        brush_value: float = 1.0,
        brush_radius_voxels: float = 2.0,
        history_depth: int = 100,
    ) -> None:
        self._data_store: ImageMemoryStore | LabelMemoryStore = data_store
        self._data_shape = data_store.shape
        self._displayed_axes: tuple[int, int] = tuple(displayed_axes)  # type: ignore[assignment]
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

    def _on_mouse_press(self, event: CanvasMousePress2DEvent) -> None:
        # Label visuals discard background fragments so they don't write to the
        # pick buffer when blank. Use a data-bounds check instead of pick
        # detection to decide whether to start a stroke.
        if not self._coord_within_data_bounds(event.world_coordinate):
            return
        self._active_stroke = ActiveStroke(
            visual_id=self._visual_id,
            data_store_id=self._data_store_id,
        )
        self._apply_brush(event.world_coordinate)

    def _coord_within_data_bounds(self, world_coord: np.ndarray) -> bool:
        """True if *world_coord* maps to a valid voxel in the data array."""
        ax_row, ax_col = self._displayed_axes
        swapped = world_coord.copy()
        swapped[ax_row], swapped[ax_col] = (
            float(world_coord[ax_col]),
            float(world_coord[ax_row]),
        )
        scene_model = self._controller._model.scenes[self._scene_id]
        visual_model = next(v for v in scene_model.visuals if v.id == self._visual_id)
        voxel_coord = visual_model.transform.imap_coordinates(np.atleast_2d(swapped))[0]
        voxel = np.round(voxel_coord).astype(np.int64)
        return all(0 <= int(voxel[i]) < self._data_shape[i] for i in range(len(voxel)))

    def _apply_brush(self, world_coord: np.ndarray) -> None:
        """Swap displayed axes before delegating to the base brush logic.

        ``CellierController._on_raw_pointer_event`` embeds the mouse position
        as ``world_coord[0] = pygfx_x``, ``world_coord[1] = pygfx_y``.
        ``GFXImageMemoryVisual`` renders ``data[r, c]`` at pygfx ``(x=c, y=r)``,
        so the row index arrives in ``world_coord[ax_col]`` and the column index
        in ``world_coord[ax_row]``.  Swapping them here produces the correct
        voxel address before the identity ``imap_coordinates`` call in the base.
        """
        ax_row, ax_col = self._displayed_axes
        swapped = world_coord.copy()
        swapped[ax_row], swapped[ax_col] = (
            float(world_coord[ax_col]),
            float(world_coord[ax_row]),
        )
        super()._apply_brush(swapped)

    def _read_old_values(self, voxel_indices: np.ndarray) -> np.ndarray:
        """Read directly from the backing numpy array."""
        idx = tuple(voxel_indices[:, i] for i in range(voxel_indices.shape[1]))
        return self._data_store.data[idx].astype(np.float32, copy=True)

    def _write_values(self, voxel_indices: np.ndarray, values: np.ndarray) -> None:
        """Write directly to the backing numpy array and reslice."""
        idx = tuple(voxel_indices[:, i] for i in range(voxel_indices.shape[1]))
        cast_values = np.round(values).astype(self._data_store.data.dtype)
        self._data_store.data[idx] = cast_values
        self._controller.reslice_scene(self._scene_id)

    def _on_stroke_completed(self, command: PaintStrokeCommand) -> None:
        """No background work needed — the array is already up to date."""

    def commit(self) -> None:
        """End the session.  Data is already in the array; just teardown."""
        self._history._undo_stack.clear()
        self._history._redo_stack.clear()
        self._teardown()

    def abort(self) -> None:
        """Revert all strokes by replaying the undo stack in reverse."""
        while self._history.can_undo:
            command = self._history.undo()
            if command is not None:
                self._write_values(command.voxel_indices, command.old_values)
        self._history._redo_stack.clear()
        self._teardown()
