from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import ConfigDict, field_serializer, field_validator

from cellier.v2.data._base_data_store import BaseDataStore

if TYPE_CHECKING:
    from cellier.v2.data.image._image_requests import ChunkRequest

_ACCEPTED_DTYPE_TYPES = {np.int8, np.int16, np.int32}


class LabelMemoryStore(BaseDataStore):
    """In-memory label data store backed by a numpy integer array.

    Serves axis-aligned slices or full sub-volumes as int32 arrays.
    Source dtype may be int8, int16, or int32; int64/uint* are rejected.

    Parameters
    ----------
    data : np.ndarray
        Integer label array (int8, int16, or int32). Shape follows numpy
        axis order — e.g. (D, H, W) for 3-D, (H, W) for 2-D.
    name : str
        Human-readable label. Default ``"label_memory_store"``.
    """

    store_type: Literal["label_memory"] = "label_memory"
    name: str = "label_memory_store"
    data: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("data", mode="before")
    @classmethod
    def _validate_integer_dtype(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v)
        if arr.dtype.type not in _ACCEPTED_DTYPE_TYPES:
            raise ValueError(
                f"LabelMemoryStore requires int8, int16, or int32 dtype. "
                f"Got {arr.dtype}. Cast your data to np.int32 first."
            )
        return np.ascontiguousarray(arr)

    @field_serializer("data")
    def _serialize_data(self, array: np.ndarray, _info: Any) -> list:
        return array.tolist()

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def n_levels(self) -> int:
        return 1

    @property
    def level_shapes(self) -> list[tuple[int, ...]]:
        return [self.shape]

    async def get_data(self, request: ChunkRequest) -> np.ndarray:
        """Return the requested sub-region as an int32 array.

        Interprets ``request.axis_selections`` generically:
        - ``int`` entry  → sliced axis (dropped from output)
        - ``(start, stop)`` tuple → displayed axis (kept in output)

        Out-of-bounds coordinates are clamped and zero-padded.
        Always returns int32 regardless of source dtype.
        """
        store_shape = self.data.shape

        out_shape: list[int] = []
        for sel in request.axis_selections:
            if isinstance(sel, tuple):
                start, stop = sel
                out_shape.append(stop - start)

        out = np.zeros(out_shape, dtype=np.int32)

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
                idx = int(np.clip(sel, 0, dim_size - 1))
                src.append(idx)

        if all_valid:
            out[tuple(dst)] = self.data[tuple(src)].astype(np.int32)

        return out
