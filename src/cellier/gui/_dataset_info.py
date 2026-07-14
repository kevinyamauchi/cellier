"""Toolkit-neutral OME-Zarr dataset metadata extraction shared by Qt and anywidget."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np

# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------


def _source_label(zarr_path: str) -> str:
    """Map a URI scheme to a human-readable source label."""
    scheme = urlparse(zarr_path).scheme
    return {
        "file": "local file",
        "s3": "S3",
        "gs": "Google Cloud Storage",
        "gcs": "Google Cloud Storage",
        "https": "HTTP",
        "http": "HTTP",
    }.get(scheme, scheme)


def _file_name(zarr_path: str) -> str:
    """Extract the final path component from a URI."""
    path = urlparse(zarr_path).path
    return path.rstrip("/").rsplit("/", 1)[-1]


@dataclass
class DatasetInfo:
    """All display-relevant metadata for one OME-Zarr dataset.

    Parameters
    ----------
    file_name :
        Basename of the zarr store path.
    zarr_type :
        OME metadata type string, e.g. ``"Image"`` or ``"Bf2Raw/Image"``.
    source :
        Human-readable storage source, e.g. ``"local file"``, ``"S3"``.
    world_to_data_matrix :
        ``(n+1, n+1)`` homogeneous affine matrix mapping physical world
        coordinates to level-0 voxel coordinates.
    axis_names :
        Axis labels in data order.
    scale_shapes :
        Shape tuple for each resolution level, finest first.
    """

    file_name: str
    zarr_type: str
    source: str
    world_to_data_matrix: np.ndarray
    axis_names: list[str]
    scale_shapes: list[tuple[int, ...]]


def _read_level_shapes(zarr_path: str, scale_paths: list[str]) -> list[tuple[int, ...]]:
    """Read per-level array shapes without importing zarr.

    Reads ``.zarray`` (zarr v2) or ``zarr.json`` (zarr v3) metadata files
    directly to avoid zarr's internal asyncio loop, which conflicts with
    a running QtAsyncio event loop.

    Supports ``file://`` URIs (reads from disk) and remote URIs (reads via
    fsspec, which yaozarrs[io] already requires).
    """
    import json

    parsed = urlparse(zarr_path)
    shapes: list[tuple[int, ...]] = []

    if parsed.scheme == "file":
        import pathlib

        root = pathlib.Path(parsed.path)
        for rel in scale_paths:
            level = root / rel
            for sentinel in (".zarray", "zarr.json"):
                meta_file = level / sentinel
                if meta_file.exists():
                    meta = json.loads(meta_file.read_text())
                    shapes.append(tuple(int(d) for d in meta["shape"]))
                    break
            else:
                raise FileNotFoundError(
                    f"No zarr metadata (.zarray or zarr.json) found at '{level}'."
                )
    else:
        import fsspec

        fs, root = fsspec.url_to_fs(zarr_path)
        for rel in scale_paths:
            level = root.rstrip("/") + "/" + rel
            for sentinel in (".zarray", "zarr.json"):
                path = level + "/" + sentinel
                if fs.exists(path):
                    with fs.open(path) as f:
                        meta = json.load(f)
                    shapes.append(tuple(int(d) for d in meta["shape"]))
                    break
            else:
                raise FileNotFoundError(
                    f"No zarr metadata (.zarray or zarr.json) found at '{level}'."
                )

    return shapes


def dataset_info_from_path(
    zarr_path: str,
    *,
    multiscale_index: int = 0,
    series_index: int = 0,
) -> DatasetInfo:
    """Extract display metadata from an OME-Zarr store.

    Uses ``yaozarrs`` to validate and read OME metadata and ``zarr`` to
    query per-level array shapes.

    Parameters
    ----------
    zarr_path :
        Root URI of the OME-Zarr group (``file://``, ``s3://``, etc.).
    multiscale_index :
        Which ``multiscales[]`` entry to read. Defaults to 0.
    series_index :
        For Bf2Raw containers, which child image series to inspect.
    """
    import yaozarrs
    from yaozarrs import v05 as ome_v05
    from yaozarrs.v05 import ScaleTransformation, TranslationTransformation

    # ── Open and validate OME metadata ──────────────────────────────────────
    group = yaozarrs.open_group(zarr_path)
    metadata = group.ome_metadata()

    zarr_type = type(metadata).__name__

    # Resolve Bf2Raw containers to a child Image.
    if isinstance(metadata, ome_v05.Bf2Raw):
        ome_subgroup = group["OME"]
        ome_meta = ome_subgroup.ome_metadata()
        if isinstance(ome_meta, ome_v05.Series):
            child_path = ome_meta.series[series_index]
            zarr_path = zarr_path.rstrip("/") + "/" + child_path
            group = yaozarrs.open_group(zarr_path)
            metadata = group.ome_metadata()
            zarr_type = f"Bf2Raw/{type(metadata).__name__}"

    if not isinstance(metadata, ome_v05.Image):
        raise TypeError(f"Expected an OME-Zarr Image, got {type(metadata).__name__!r}.")

    ms = metadata.multiscales[multiscale_index]

    # ── Axis names ──────────────────────────────────────────────────────────
    axis_names = [ax.name for ax in ms.axes]
    n = len(axis_names)

    # ── Global coordinate transforms ────────────────────────────────────────
    global_scale = [1.0] * n
    global_translation = [0.0] * n

    if ms.coordinateTransformations is not None:
        for ct in ms.coordinateTransformations:
            if isinstance(ct, ScaleTransformation):
                global_scale = list(ct.scale)
            elif isinstance(ct, TranslationTransformation):
                global_translation = list(ct.translation)

    # Level-0 dataset scale.
    ds0 = ms.datasets[0]
    ds0_scale = list(ds0.scale_transform.scale)

    # Physical scale per axis at level 0: global_scale * dataset0_scale.
    phys_scale = [global_scale[i] * ds0_scale[i] for i in range(n)]

    # ── World-to-data matrix (homogeneous, (n+1) x (n+1)) ──────────────────
    # data-to-world: world[i] = phys_scale[i] * voxel[i] + global_translation[i]
    # world-to-data: voxel[i] = (world[i] - global_translation[i]) / phys_scale[i]
    mat = np.zeros((n + 1, n + 1), dtype=float)
    for i in range(n):
        s = phys_scale[i] if phys_scale[i] != 0.0 else 1.0
        mat[i, i] = 1.0 / s
        mat[i, n] = -global_translation[i] / s
    mat[n, n] = 1.0

    # ── Per-level shapes via direct JSON reads ──────────────────────────────
    # Avoid zarr.open_group: zarr v3 tries to start an asyncio event loop,
    # which raises RuntimeError when QtAsyncio is already running.
    scale_paths = [ds.path for ds in ms.datasets]
    scale_shapes = _read_level_shapes(zarr_path, scale_paths)

    return DatasetInfo(
        file_name=_file_name(zarr_path),
        zarr_type=zarr_type,
        source=_source_label(zarr_path),
        world_to_data_matrix=mat,
        axis_names=axis_names,
        scale_shapes=scale_shapes,
    )
