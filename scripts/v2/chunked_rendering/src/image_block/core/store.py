"""Tensorstore zarr I/O helpers shared by 2D and 3D pipelines."""

from __future__ import annotations

import pathlib

import tensorstore as ts


def _detect_zarr_driver(level_path: pathlib.Path) -> str:
    """Return the correct tensorstore driver for a zarr level directory.

    Detects format by the metadata sentinel file:
    - ``.zarray``   -> zarr v2  -> driver ``"zarr"``
    - ``zarr.json`` -> zarr v3  -> driver ``"zarr3"``
    """
    p = pathlib.Path(level_path)
    if (p / ".zarray").exists():
        return "zarr"
    if (p / "zarr.json").exists():
        return "zarr3"
    raise FileNotFoundError(
        f"Cannot determine zarr format for '{p}': "
        f"neither '.zarray' (zarr v2) nor 'zarr.json' (zarr v3) found.\n"
        f"Re-run with --make-files to regenerate the store."
    )


def open_ts_stores(
    zarr_path: pathlib.Path,
    scale_names: list[str],
) -> list[ts.TensorStore]:
    """Open one tensorstore per scale level (read-only, synchronous).

    Auto-detects zarr v2 vs v3 format.
    Must be called **before** ``QtAsyncio.run()`` starts the event loop.

    Parameters
    ----------
    zarr_path : pathlib.Path
        Root directory of the multiscale zarr store.
    scale_names : list[str]
        Subdirectory names in order finest -> coarsest.

    Returns
    -------
    stores : list[ts.TensorStore]
        One open store per scale level.
    """
    stores = []
    for name in scale_names:
        level_path = pathlib.Path(zarr_path) / name
        driver = _detect_zarr_driver(level_path)
        spec = {
            "driver": driver,
            "kvstore": {
                "driver": "file",
                "path": str(level_path),
            },
        }
        store = ts.open(spec).result()
        shape_str = tuple(int(d) for d in store.domain.shape)
        fmt = "v2" if driver == "zarr" else "v3"
        print(f"  {name}: opened as zarr {fmt}  shape={shape_str}")
        stores.append(store)
    return stores
