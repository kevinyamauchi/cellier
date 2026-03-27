# src/cellier/v2/data/image/_image_memory_store.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import ConfigDict, field_serializer, field_validator

from cellier.v2.data._base_data_store import BaseDataStore

if TYPE_CHECKING:
    from cellier.v2.data.image._image_requests import ChunkRequest


class ImageMemoryStore(BaseDataStore):
    """In-memory image data store backed by a numpy array.

    Serves axis-aligned slices or full sub-volumes to the AsyncSlicer.
    All reads are synchronous (the array is in CPU RAM); the method is
    still declared ``async`` to satisfy the AsyncSlicer contract.

    Parameters
    ----------
    data : np.ndarray
        The image data. Any dtype; coerced to float32 on construction.
        Shape convention follows numpy axis order — e.g. (D, H, W) for
        3-D, (H, W) for 2-D, (T, C, D, H, W) for 5-D.
    name : str
        Human-readable label. Default ``"image_memory_store"``.
    """

    store_type: Literal["image_memory"] = "image_memory"
    name: str = "image_memory_store"
    data: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ------------------------------------------------------------------
    # Pydantic validators / serializers for numpy
    # ------------------------------------------------------------------

    @field_validator("data", mode="before")
    @classmethod
    def _coerce_float32(cls, v: Any) -> np.ndarray:
        """Coerce the input to a contiguous float32 C-array."""
        arr = np.asarray(v, dtype=np.float32)
        return np.ascontiguousarray(arr)

    @field_serializer("data")
    def _serialize_data(self, array: np.ndarray, _info: Any) -> list:
        """Serialise the array as a nested Python list for JSON round-trips."""
        return array.tolist()

    # ------------------------------------------------------------------
    # Read-only properties (used by CellierController.add_image)
    # ------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Number of dimensions in the stored array."""
        return self.data.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the stored array in numpy axis order."""
        return tuple(self.data.shape)

    @property
    def n_levels(self) -> int:
        """Always 1 — single-resolution, no multiscale pyramid."""
        return 1

    @property
    def level_shapes(self) -> list[tuple[int, ...]]:
        """List with one entry (level 0 = the full array)."""
        return [self.shape]

    # ------------------------------------------------------------------
    # Async data access (called by AsyncSlicer)
    # ------------------------------------------------------------------

    async def get_data(self, request: ChunkRequest) -> np.ndarray:
        """Return the requested sub-region as a float32 array.

        Interprets ``request.axis_selections`` generically:

        - ``int`` entry  → sliced axis; the integer index is applied and the
          axis is dropped from the output.
        - ``(start, stop)`` tuple → displayed axis; a slice is applied and
          the axis is kept in the output.

        Out-of-bounds coordinates are clamped to array extents and
        zero-padded on the output side so the returned shape always matches
        what the caller requested.

        Parameters
        ----------
        request : ChunkRequest
            Built by ``GFXImageMemoryVisual.build_slice_request[_2d]``.
            ``request.scale_index`` is always 0 (ignored).
            ``request.axis_selections`` has one entry per data axis.

        Returns
        -------
        np.ndarray
            float32 array with one dimension per displayed (tuple) axis.
        """
        store_shape = self.data.shape

        # ── 1. Compute the output shape ─────────────────────────────────
        out_shape: list[int] = []
        for sel in request.axis_selections:
            if isinstance(sel, tuple):
                start, stop = sel
                out_shape.append(stop - start)
            # int → axis dropped from output

        out = np.zeros(out_shape, dtype=np.float32)

        # ── 2. Build clamped source indices and destination slices ───────
        src: list[int | slice] = []
        dst: list[slice] = []
        all_valid = True

        for ax, sel in enumerate(request.axis_selections):
            dim_size = store_shape[ax]
            if isinstance(sel, tuple):
                start, stop = sel
                c_start = max(0, start)
                c_stop = min(dim_size, stop)
                if c_stop <= c_start:
                    all_valid = False
                    break
                src.append(slice(c_start, c_stop))
                dst_start = c_start - start
                dst_stop = dst_start + (c_stop - c_start)
                dst.append(slice(dst_start, dst_stop))
            else:
                # Scalar — clamp to valid range, keeps axis out of output.
                idx = int(np.clip(sel, 0, dim_size - 1))
                src.append(idx)

        if all_valid:
            out[tuple(dst)] = self.data[tuple(src)]

        return out
