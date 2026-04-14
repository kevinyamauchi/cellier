"""OMEZarrImageDataStore — data store for OME-Zarr v0.5 images.

Reads validated OME metadata via ``yaozarrs`` and opens per-level
tensorstore handles for async data access.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

import numpy as np
import tensorstore as ts
from pydantic import ConfigDict, PrivateAttr

from cellier.v2.data._base_data_store import BaseDataStore
from cellier.v2.data.image._axis_info import AxisInfo
from cellier.v2.transform import AffineTransform

if TYPE_CHECKING:
    from yaozarrs import v05

    from cellier.v2.data.image._image_requests import ChunkRequest

# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------

_SUPPORTED_SCHEMES = frozenset({"file", "s3", "gs", "gcs", "https", "http"})


def _validate_uri_scheme(uri: str) -> None:
    """Raise ``ValueError`` if *uri* does not start with a supported scheme."""
    parsed = urlparse(uri)
    if parsed.scheme not in _SUPPORTED_SCHEMES:
        raise ValueError(
            f"Unsupported URI scheme {parsed.scheme!r} in {uri!r}. "
            f"Supported schemes: {', '.join(sorted(_SUPPORTED_SCHEMES))}."
        )


def _build_kvstore_spec(uri: str, array_path: str) -> dict:
    """Build a TensorStore kvstore spec from root URI and relative array path.

    Supports ``file://``, ``s3://``, ``gs://`` / ``gcs://``,
    ``http://`` and ``https://``.
    """
    parsed = urlparse(uri)
    scheme = parsed.scheme

    if scheme == "file":
        # file:///absolute/path  ->  path = /absolute/path
        root = parsed.path
        return {
            "driver": "file",
            "path": str(pathlib.PurePosixPath(root) / array_path),
        }
    elif scheme in ("s3",):
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        full = f"{prefix}/{array_path}" if prefix else array_path
        return {"driver": "s3", "bucket": bucket, "path": full}
    elif scheme in ("gs", "gcs"):
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        full = f"{prefix}/{array_path}" if prefix else array_path
        return {"driver": "gcs", "bucket": bucket, "path": full}
    elif scheme in ("http", "https"):
        base = uri.rstrip("/")
        return {"driver": "http", "base_url": f"{base}/{array_path}"}
    else:
        raise ValueError(f"Unsupported URI scheme: {scheme!r}")


# ---------------------------------------------------------------------------
# TensorStore opening
# ---------------------------------------------------------------------------


def _detect_zarr_driver(uri: str, array_path: str) -> str:
    """Detect zarr format (v2 or v3) for a level inside a URI.

    For ``file://`` URIs, checks sentinel files on disk.
    For remote URIs, defaults to ``zarr3``.
    """
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        if (len(parsed.netloc) > 0) and (len(parsed.path) == 0):
            level_path = pathlib.Path(parsed.netloc) / array_path
        elif (len(parsed.netloc) == 0) and (len(parsed.path) > 0):
            level_path = pathlib.Path(parsed.path) / array_path
        else:
            raise ValueError(f"URI couldn't be parsed: {parsed}")

        if (level_path / ".zarray").exists():
            return "zarr"
        if (level_path / "zarr.json").exists():
            return "zarr3"
        raise FileNotFoundError(
            f"Cannot determine zarr format for '{level_path}': "
            f"{array_path} is not a zarr file."
            f"{parsed} didn't work"
            f"neither '.zarray' (zarr v2) nor 'zarr.json' (zarr v3) found."
        )
    # Remote: default to zarr3 (OME-Zarr v0.5 implies zarr v3).
    return "zarr3"


def _open_ome_ts_stores(
    zarr_path: str,
    scale_names: list[str],
    anonymous: bool = False,
) -> list[ts.TensorStore]:
    """Open one TensorStore per scale level (synchronous, read-only).

    Must be called before ``QtAsyncio.run()`` starts the event loop.

    Parameters
    ----------
    zarr_path : str
        Root URI of the OME-Zarr store.
    scale_names : list[str]
        Per-level relative array paths.
    anonymous : bool
        When True, use anonymous credentials for S3/GCS access
        (for public buckets). Default False.
    """
    stores: list[ts.TensorStore] = []
    scheme = urlparse(zarr_path).scheme
    for name in scale_names:
        driver = _detect_zarr_driver(zarr_path, name)
        spec: dict[str, Any] = {
            "driver": driver,
            "kvstore": _build_kvstore_spec(zarr_path, name),
        }
        # Use anonymous credentials for public cloud buckets.
        if anonymous and scheme in ("s3", "gs", "gcs"):
            if scheme == "s3":
                spec.setdefault("context", {})["aws_credentials"] = {
                    "anonymous": True,
                }
            else:
                spec.setdefault("context", {})["gcs_user_project"] = ""
        store = ts.open(spec).result()
        shape = tuple(int(d) for d in store.domain.shape)
        print(f"  {name}: opened as {driver}  shape={shape}")
        stores.append(store)
    return stores


# ---------------------------------------------------------------------------
# OME metadata helpers
# ---------------------------------------------------------------------------


def _extract_global_transform(
    ms: v05.Multiscale,
) -> tuple[list[float], list[float]]:
    """Extract global scale and translation from ``ms.coordinateTransformations``.

    Returns identity values when the field is absent.
    """
    from yaozarrs.v05 import ScaleTransformation, TranslationTransformation

    n = len(ms.axes)
    global_scale = [1.0] * n
    global_translation = [0.0] * n

    if ms.coordinateTransformations is not None:
        for ct in ms.coordinateTransformations:
            if isinstance(ct, ScaleTransformation):
                global_scale = list(ct.scale)
            elif isinstance(ct, TranslationTransformation):
                global_translation = list(ct.translation)

    return global_scale, global_translation


def _derive_level_transforms(
    ms: v05.Multiscale,
    global_scale: list[float],
    global_translation: list[float],
) -> list[AffineTransform]:
    """Compute per-level voxel-level-k → voxel-level-0 AffineTransforms.

    Implements the math from §3.2 of the design document over all axes.
    """
    n_axes = len(ms.axes)

    per_level: list[tuple[list[float], list[float]]] = []
    for ds in ms.datasets:
        sc = [g * s for g, s in zip(global_scale, ds.scale_transform.scale)]
        tr_raw = ds.translation_transform
        tr_ds = list(tr_raw.translation) if tr_raw is not None else [0.0] * n_axes
        tr = [gs * t + gt for gs, t, gt in zip(global_scale, tr_ds, global_translation)]
        per_level.append((sc, tr))

    s0, t0 = per_level[0]
    transforms: list[AffineTransform] = []
    for sc_k, tr_k in per_level:
        cellier_scale = tuple(sc_k[i] / s0[i] for i in range(n_axes))
        cellier_trans = tuple((tr_k[i] - t0[i]) / s0[i] for i in range(n_axes))
        transforms.append(
            AffineTransform.from_scale_and_translation(
                scale=cellier_scale, translation=cellier_trans
            )
        )
    return transforms


# ---------------------------------------------------------------------------
# OMEZarrImageDataStore
# ---------------------------------------------------------------------------


class OMEZarrImageDataStore(BaseDataStore):
    """Data store for an OME-Zarr v0.5 image read via tensorstore.

    Use the :meth:`from_path` class method to construct from an OME-Zarr URI.

    Parameters
    ----------
    store_type : Literal["ome_zarr_image"]
        Discriminator field. Always ``"ome_zarr_image"``.
    zarr_path : str
        URI to the root OME-Zarr group. Must start with ``file://``,
        ``s3://``, ``gs://``, or ``https://``.
    multiscale_index : int
        Index into ``multiscales[]``. Defaults to 0.
    scale_names : list[str]
        Per-level relative array paths, finest to coarsest.
    level_transforms : list[AffineTransform]
        Full-rank (all axes) AffineTransform per level:
        voxel-level-k to voxel-level-0.
    axis_names : list[str]
        All axis names in data order.
    axis_units : list[str | None]
        Physical units per axis (``None`` if unspecified).
    axis_types : list[str]
        OME axis type per axis.
    name : str
        Human-readable name for the store.
    """

    store_type: Literal["ome_zarr_image"] = "ome_zarr_image"
    zarr_path: str
    multiscale_index: int = 0
    scale_names: list[str]
    level_transforms: list[AffineTransform]
    axis_names: list[str]
    axis_units: list[str | None]
    axis_types: list[str]
    anonymous: bool = False
    name: str = "ome zarr image data store"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _ts_stores: list[ts.TensorStore] = PrivateAttr(default_factory=list)

    # ── Lifecycle ───────────────────────────────────────────────────────

    def model_post_init(self, __context: Any) -> None:
        """Open all TensorStore handles (synchronous, before QtAsyncio)."""
        self._ts_stores = _open_ome_ts_stores(
            self.zarr_path, self.scale_names, anonymous=self.anonymous
        )

    # ── Convenience constructor ─────────────────────────────────────────

    @classmethod
    def from_path(
        cls,
        zarr_path: str,
        *,
        multiscale_index: int = 0,
        series_index: int = 0,
        anonymous: bool = False,
        name: str = "ome zarr image data store",
    ) -> OMEZarrImageDataStore:
        """Construct from an OME-Zarr v0.5 URI.

        Supports both standard Image stores and Bf2Raw (bioformats2raw)
        multi-series containers.  For Bf2Raw stores the *series_index*
        selects which child image to open.

        Parameters
        ----------
        zarr_path : str
            URI with a scheme prefix: ``file://``, ``s3://``, ``gs://``,
            or ``https://``.  For local files use an absolute path, e.g.
            ``file:///home/user/data/image.ome.zarr``.
        multiscale_index : int
            Which ``multiscales[]`` entry to use. Defaults to 0.
        series_index : int
            For Bf2Raw containers, which image series to open.
            Ignored for standard Image stores. Defaults to 0.
        anonymous : bool
            When True, use anonymous credentials for S3/GCS access
            (for public buckets). Default False.
        name : str
            Human-readable name for the store.

        Raises
        ------
        ValueError
            If the URI scheme is not supported or the series index is
            out of range.
        TypeError
            If the OME metadata is neither Image nor Bf2Raw (e.g. a Plate).
        """
        import yaozarrs
        from yaozarrs import v05 as ome_v05

        # 1. Validate URI scheme.
        _validate_uri_scheme(zarr_path)

        # 2. Open and validate OME metadata via yaozarrs.
        group = yaozarrs.open_group(zarr_path)
        metadata = group.ome_metadata()

        # 3. Handle Bf2Raw containers: navigate to the child image group.
        if isinstance(metadata, ome_v05.Bf2Raw):
            zarr_path, group, metadata = cls._resolve_bf2raw(
                zarr_path, group, series_index
            )

        # 4. Check it is an Image.
        if not isinstance(metadata, ome_v05.Image):
            type_name = type(metadata).__name__ if metadata is not None else "None"
            raise TypeError(
                f"Expected an OME-Zarr Image at {zarr_path!r}, "
                f"got {type_name}. Plates and other types are not supported."
            )

        # 5. Select multiscale entry.
        ms = metadata.multiscales[multiscale_index]

        # 6. Collect axis metadata.
        axis_names = [ax.name for ax in ms.axes]
        axis_units = [getattr(ax, "unit", None) for ax in ms.axes]
        axis_types = [ax.type or "" for ax in ms.axes]

        # 7. Extract global coordinateTransformations.
        global_scale, global_translation = _extract_global_transform(ms)

        # 8. Derive per-level AffineTransforms (full rank, all axes).
        level_transforms = _derive_level_transforms(
            ms, global_scale, global_translation
        )

        # 9. Collect scale_names.
        scale_names = [ds.path for ds in ms.datasets]

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

    @classmethod
    def _resolve_bf2raw(
        cls,
        zarr_path: str,
        group: Any,
        series_index: int,
    ) -> tuple[str, Any, Any]:
        """Navigate from a Bf2Raw root to the requested child image group.

        Parameters
        ----------
        zarr_path : str
            Root URI of the Bf2Raw container.
        group : yaozarrs.ZarrGroup
            Already-opened root group.
        series_index : int
            Which child image to open.

        Returns
        -------
        resolved_path : str
            URI pointing at the child image group.
        child_group : yaozarrs.ZarrGroup
            Opened child group.
        child_metadata : v05.Image | v05.OMEMetadata
            OME metadata read from the child group.

        Raises
        ------
        ValueError
            If *series_index* is out of range.
        """
        import yaozarrs
        from yaozarrs import v05 as ome_v05

        # Try to read the OME/zarr.json Series list for canonical paths.
        series_paths: list[str] | None = None
        if "OME" in group:
            ome_subgroup = group["OME"]
            ome_meta = ome_subgroup.ome_metadata()
            if isinstance(ome_meta, ome_v05.Series):
                series_paths = ome_meta.series

        if series_paths is not None:
            if series_index < 0 or series_index >= len(series_paths):
                raise ValueError(
                    f"series_index={series_index} out of range: "
                    f"Bf2Raw container has {len(series_paths)} series "
                    f"({series_paths})."
                )
            image_path = series_paths[series_index]
        else:
            # No Series metadata — fall back to numeric path.
            image_path = str(series_index)

        resolved_path = zarr_path.rstrip("/") + "/" + image_path
        print(
            f"  Bf2Raw container detected — resolving series {series_index} "
            f"at '{image_path}'"
        )

        child_group = yaozarrs.open_group(resolved_path)
        child_metadata = child_group.ome_metadata()

        return resolved_path, child_group, child_metadata

    # ── Read-only properties ────────────────────────────────────────────

    @property
    def n_levels(self) -> int:
        """Number of scale levels."""
        return len(self._ts_stores)

    @property
    def level_shapes(self) -> list[tuple[int, ...]]:
        """Full-rank shape per level (all axes), finest first.

        Returns shapes over all axes, including non-spatial ones.
        The controller projects to the displayed subshape using
        ``dims.displayed_axes`` before constructing the render visual.
        """
        return [tuple(int(d) for d in store.domain.shape) for store in self._ts_stores]

    @property
    def axes(self) -> list[AxisInfo]:
        """All axes in data order as AxisInfo descriptors.

        Use ``array_dim`` and ``type`` to configure ``dims.displayed_axes``
        and ``dims.selection.slice_indices`` before rendering. Example::

            scene.dims.displayed_axes = [
                ax.array_dim for ax in store.axes if ax.type == "space"
            ]
            scene.dims.selection.slice_indices = {
                ax.array_dim: 0 for ax in store.axes if ax.type != "space"
            }
        """
        return [
            AxisInfo(name=n, unit=u, type=t, array_dim=i)
            for i, (n, u, t) in enumerate(
                zip(self.axis_names, self.axis_units, self.axis_types)
            )
        ]

    @property
    def dtype(self) -> np.dtype:
        """Data type of the underlying arrays."""
        return self._ts_stores[0].dtype.numpy_dtype

    # ── Async data access ───────────────────────────────────────────────

    async def get_data(self, request: ChunkRequest) -> np.ndarray:
        """Read a single padded brick, returning a zero-padded float32 array.

        Interprets ``request.axis_selections`` generically: displayed axes
        (tuple ranges) become slice dimensions in the output; sliced axes
        (int values) become point selections.

        Parameters
        ----------
        request : ChunkRequest
            Padded brick specification.

        Returns
        -------
        np.ndarray
            ``float32`` array.
        """
        store = self._ts_stores[request.scale_index]
        store_shape = tuple(int(d) for d in store.domain.shape)

        out_shape = tuple(
            stop - start
            for sel in request.axis_selections
            if isinstance(sel, tuple)
            for start, stop in [sel]
        )
        out = np.zeros(out_shape, dtype=np.float32)

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
                dtype=np.float32,
            )
            dest_idx = tuple(slice(d, d + s) for d, s in zip(dest_starts, region.shape))
            out[dest_idx] = region

        return out
