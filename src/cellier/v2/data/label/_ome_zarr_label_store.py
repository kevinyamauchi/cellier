"""OMEZarrLabelDataStore — data store for OME-Zarr v0.5 label images.

Reads OME-NGFF label groups and opens per-level tensorstore handles
for async data access.  Returns int32 bricks (never float32).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import ConfigDict, PrivateAttr

if TYPE_CHECKING:
    import tensorstore as ts

from cellier.v2.data._base_data_store import BaseDataStore
from cellier.v2.data.image._ome_zarr_image_store import _validate_uri_scheme
from cellier.v2.transform import AffineTransform

_ACCEPTED_LABEL_DTYPES = {np.int8, np.int16, np.int32}


class OMEZarrLabelDataStore(BaseDataStore):
    """Multiscale OME-Zarr label store returning int32 bricks.

    Use the :meth:`from_path` class method to construct from a URI that
    points to an OME-NGFF label group (containing ``ome`` → ``multiscales``
    metadata).

    Parameters
    ----------
    store_type : Literal["ome_zarr_label"]
        Discriminator field. Always ``"ome_zarr_label"``.
    zarr_path : str
        URI to the label group.  Must point to the label sub-group
        (e.g. ``"file:///path/to/seg.ome.zarr/labels/cells"``),
        not the root OME-Zarr.
    multiscale_index : int
        Which multiscale entry to use (default 0).
    scale_names : list[str]
        Per-level relative array paths, finest to coarsest.
    level_transforms : list[AffineTransform]
        Per-level voxel-level-k → voxel-level-0 transforms.
    axis_names : list[str]
        All axis names in data order.
    axis_units : list[str | None]
        Physical units per axis (``None`` if unspecified).
    axis_types : list[str]
        OME axis type per axis.
    name : str
        Human-readable name for the store.
    """

    store_type: Literal["ome_zarr_label"] = "ome_zarr_label"
    zarr_path: str
    multiscale_index: int = 0
    scale_names: list[str]
    level_transforms: list[AffineTransform]
    axis_names: list[str]
    axis_units: list[str | None]
    axis_types: list[str]
    anonymous: bool = False
    name: str = "ome zarr label data store"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _ts_stores: list[ts.TensorStore] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Open all TensorStore handles (synchronous, before QtAsyncio)."""
        from cellier.v2.data.image._ome_zarr_image_store import _open_ome_ts_stores

        self._ts_stores = _open_ome_ts_stores(
            self.zarr_path, self.scale_names, anonymous=self.anonymous
        )

    # ── Convenience constructors ────────────────────────────────────────

    @classmethod
    def from_path(
        cls,
        zarr_path: str,
        *,
        multiscale_index: int = 0,
        anonymous: bool = False,
        name: str = "ome zarr label data store",
    ) -> OMEZarrLabelDataStore:
        """Construct from a URI pointing directly at an OME-NGFF label group.

        The URI must point at a zarr group that carries ``ome.multiscales``
        metadata (i.e. the label sub-group itself, not the root OME-Zarr).

        Parameters
        ----------
        zarr_path : str
            URI with a scheme prefix: ``file://``, ``s3://``, ``gs://``,
            or ``https://``.  For local files use an absolute path, e.g.
            ``file:///home/user/data/seg.ome.zarr/labels/cells``.
        multiscale_index : int
            Which ``multiscales[]`` entry to use. Defaults to 0.
        anonymous : bool
            When True, use anonymous credentials for S3/GCS access.
        name : str
            Human-readable name for the store.
        """
        import yaozarrs

        _validate_uri_scheme(zarr_path)

        group = yaozarrs.open_group(zarr_path)
        raw_attrs = group.attrs

        # OME-NGFF label groups store metadata under "ome" → "multiscales"
        # (as written by the demo writer) or at the top-level "multiscales".
        multiscales_list = cls._read_multiscales(raw_attrs, zarr_path)
        ms_raw = multiscales_list[multiscale_index]
        ms = cls._parse_multiscale(ms_raw)

        axis_names = [ax["name"] for ax in ms["axes"]]
        axis_units = [ax.get("unit") for ax in ms["axes"]]
        axis_types = [ax.get("type", "") for ax in ms["axes"]]

        n_axes = len(axis_names)
        global_scale = [1.0] * n_axes
        global_translation = [0.0] * n_axes
        for ct in ms.get("coordinateTransformations") or []:
            if ct.get("type") == "scale":
                global_scale = list(ct["scale"])
            elif ct.get("type") == "translation":
                global_translation = list(ct["translation"])

        datasets = ms["datasets"]
        scale_names = [ds["path"] for ds in datasets]

        # Build per-level AffineTransforms manually (labels use raw dicts,
        # not yaozarrs typed objects).
        per_level: list[tuple[list[float], list[float]]] = []
        for ds in datasets:
            cts = ds.get("coordinateTransformations") or []
            ds_scale = [1.0] * n_axes
            ds_trans = [0.0] * n_axes
            for ct in cts:
                if ct.get("type") == "scale":
                    ds_scale = list(ct["scale"])
                elif ct.get("type") == "translation":
                    ds_trans = list(ct["translation"])
            sc = [g * s for g, s in zip(global_scale, ds_scale)]
            tr = [
                gs * t + gt
                for gs, t, gt in zip(global_scale, ds_trans, global_translation)
            ]
            per_level.append((sc, tr))

        s0, t0 = per_level[0]
        level_transforms: list[AffineTransform] = []
        for sc_k, tr_k in per_level:
            cellier_scale = tuple(sc_k[i] / s0[i] for i in range(n_axes))
            cellier_trans = tuple((tr_k[i] - t0[i]) / s0[i] for i in range(n_axes))
            level_transforms.append(
                AffineTransform.from_scale_and_translation(
                    scale=cellier_scale, translation=cellier_trans
                )
            )

        return cls(
            zarr_path=zarr_path,
            multiscale_index=multiscale_index,
            scale_names=scale_names,
            level_transforms=level_transforms,
            axis_names=axis_names,
            axis_units=axis_units,
            axis_types=axis_types,
            anonymous=anonymous,
            name=name,
        )

    @staticmethod
    def _read_multiscales(attrs: dict, zarr_path: str) -> list:
        """Extract multiscales list from raw zarr attributes dict."""
        # OME-NGFF label groups may store metadata under "ome.multiscales"
        if "ome" in attrs and isinstance(attrs["ome"], dict):
            ome_block = attrs["ome"]
            if "multiscales" in ome_block:
                return ome_block["multiscales"]
        # Or at the top-level "multiscales" (older convention)
        if "multiscales" in attrs:
            return attrs["multiscales"]
        raise ValueError(
            f"No multiscales metadata found at {zarr_path!r}. "
            f"Expected 'ome.multiscales' or 'multiscales' in zarr attributes."
        )

    @staticmethod
    def _parse_multiscale(ms_raw: dict) -> dict:
        """Normalize a multiscale metadata dict."""
        return ms_raw

    # ── Read-only properties ────────────────────────────────────────────

    @property
    def n_levels(self) -> int:
        """Number of scale levels."""
        return len(self._ts_stores)

    @property
    def level_shapes(self) -> list[tuple[int, ...]]:
        """Full-rank shape per level (all axes), finest first."""
        return [tuple(int(d) for d in store.domain.shape) for store in self._ts_stores]

    @property
    def dtype(self) -> np.dtype:
        """Data type of the underlying arrays (must be int8/int16/int32)."""
        native = self._ts_stores[0].dtype.numpy_dtype
        if native.type not in _ACCEPTED_LABEL_DTYPES:
            raise ValueError(
                f"OMEZarrLabelDataStore: unsupported dtype {native}. "
                f"Expected int8, int16, or int32."
            )
        return native

    @property
    def ndim(self) -> int:
        """Number of data dimensions."""
        return len(self.axis_names)

    # ── Async data access ───────────────────────────────────────────────

    async def get_data(self, request) -> np.ndarray:
        """Read a padded brick, returning int32 (zero-padded for out-of-bounds).

        Parameters
        ----------
        request : ChunkRequest
            Padded brick specification with ``axis_selections`` and
            ``scale_index``.

        Returns
        -------
        np.ndarray
            int32 array.
        """
        store = self._ts_stores[request.scale_index]
        store_shape = tuple(int(d) for d in store.domain.shape)

        out_shape = tuple(
            stop - start
            for sel in request.axis_selections
            if isinstance(sel, tuple)
            for start, stop in [sel]
        )
        out = np.zeros(out_shape, dtype=np.int32)

        store_idx: list[int | slice] = []
        dest_starts: list[int] = []
        valid = True

        for axis_i, sel in enumerate(request.axis_selections):
            size = store_shape[axis_i]
            if isinstance(sel, int):
                store_idx.append(max(0, min(sel, size - 1)))
            else:
                start, stop = sel
                c_start = max(start, 0)
                c_stop = min(stop, size)
                if c_stop <= c_start:
                    valid = False
                    break
                store_idx.append(slice(c_start, c_stop))
                dest_starts.append(c_start - start)

        if valid:
            region = np.asarray(
                await store[tuple(store_idx)].read(),
                dtype=np.int32,
            )
            dest_idx = tuple(slice(d, d + s) for d, s in zip(dest_starts, region.shape))
            out[dest_idx] = region

        return out
