"""Abstract base class providing the shared async commit loop."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from image_block.core.block_key import BlockKey
from image_block.core.cache import commit_block
from image_block.core.tile_manager import TileSlot


class BlockStateBase(ABC):
    """Abstract base for multi-LOAD block-cache state (2D or 3D).

    Subclasses must implement:
      _read_block_async(key: BlockKey) -> np.ndarray
      _rebuild_lut() -> None

    The async commit loop is implemented once here and shared by both
    BlockState2D and BlockState3D.

    Subclass __init__ must set these instance attributes before any
    commit loop is called:
      cache_info, cache_data, cache_tex, tile_manager
    """

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement both methods
    # ------------------------------------------------------------------

    @abstractmethod
    async def _read_block_async(self, key: BlockKey) -> np.ndarray:
        """Read one padded block from the backing tensorstore.

        Must return a float32 ndarray of shape:
          2D: (padded_bs, padded_bs)
          3D: (padded_bs, padded_bs, padded_bs)
        """
        ...

    @abstractmethod
    def _rebuild_lut(self) -> None:
        """Rewrite lut_data from tile_manager.tilemap and upload to GPU."""
        ...

    # ------------------------------------------------------------------
    # Shared async commit loop
    # ------------------------------------------------------------------

    async def _commit_blocks_async(
        self,
        fill_plan: list[tuple[BlockKey, TileSlot]],
        status_callback: Callable[[str], None] | None = None,
        batch_size: int = 8,
        label: str = "blocks",
    ) -> None:
        """Batched async commit loop shared by BlockState2D and BlockState3D.

        Per batch:
        1. asyncio.gather all reads concurrently (tensorstore pipelines I/O).
        2. commit_block each result into the CPU cache + schedule update_range.
        3. _rebuild_lut() once for the whole batch.
        4. Optional status_callback.
        5. asyncio.sleep(0) to yield to Qt — renderer flushes and redraws.

        Re-raises CancelledError so asyncio marks the task properly cancelled.

        Parameters
        ----------
        fill_plan : list[tuple[BlockKey, TileSlot]]
            Ordered nearest-first.  Produced by plan_update().
        status_callback : callable or None
            Optional f(text: str) -> None.  Called after each batch.
        batch_size : int
            Blocks per batch before yielding.
        label : str
            Word used in status messages, e.g. "tiles" or "bricks".
        """
        arrived = 0
        total = len(fill_plan)
        batches = [fill_plan[i : i + batch_size] for i in range(0, total, batch_size)]

        try:
            for batch in batches:
                # Step 1: concurrent reads.
                results = await asyncio.gather(
                    *[self._read_block_async(key) for key, _slot in batch]
                )

                # Step 2: commit into CPU cache.
                for (key, slot), data in zip(batch, results, strict=False):
                    commit_block(
                        self.cache_data,
                        self.cache_tex,
                        slot.grid_pos,
                        self.cache_info.padded_block_size,
                        data,
                    )
                    arrived += 1

                # Step 3: rebuild LUT once per batch.
                self._rebuild_lut()

                # Step 4: status update.
                if status_callback is not None:
                    status_callback(f"Loading: {arrived} / {total} {label}")

                # Step 5: yield to Qt.
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            print(f"  commit cancelled after {arrived}/{total} {label}")
            if status_callback is not None:
                status_callback(f"Cancelled ({arrived}/{total} {label})")
            raise

        print(
            f"  commit complete: {arrived}/{total} {label}  "
            f"({len(batches)} batches of up to {batch_size})"
        )
        if status_callback is not None:
            status_callback(f"Ready  ({arrived} {label} loaded)")
