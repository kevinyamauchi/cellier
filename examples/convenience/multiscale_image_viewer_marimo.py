"""Multiscale image viewer marimo app."""

import marimo

__generated_with = "0.23.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multiscale image viewer

    A multiscale image viewer demonstrating all appearance widgets:

    - **Colormap** -- choose from ten scientific colormaps
    - **Contrast limits** -- drag the lo and hi handles independently
    - **Render mode** -- ISO surface / MIP / attenuated MIP
    - **ISO threshold** -- visible in ISO mode; controls the isosurface level
    - **Attenuation** -- visible in attenuated MIP mode; controls depth darkening
    - **LOD bias** -- coarsen or refine level-of-detail (most visible when zoomed out)
    - **Bounding box** -- enable, line width, and color
    - **Dataset info** -- expand the detail block for array metadata

    **Interaction (3D):** orbit (left-drag), zoom (scroll), pan (right-drag)

    **Interaction (2D):** pan (left-drag), zoom (scroll), slice (z slider)

    Note that this requires installing render canvas from main and tensorstore:

    ```bash
    uv pip install git+https://github.com/pygfx/rendercanvas.git tensorstore
    ```
    """)
    return


@app.cell
def _():
    import tempfile
    from pathlib import Path

    import numpy as np
    import tensorstore as ts

    from cellier.convenience import (
        AppearanceControls,
        Layout,
        SceneControls,
        Viewer,
        axis_ranges_from_viewer,
        display,
    )
    from cellier.convenience.gui import build_canvas_widget
    from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
    from cellier.transform import AffineTransform

    return (
        AffineTransform,
        AppearanceControls,
        Layout,
        MultiscaleZarrDataStore,
        Path,
        SceneControls,
        Viewer,
        axis_ranges_from_viewer,
        build_canvas_widget,
        display,
        np,
        tempfile,
        ts,
    )


@app.cell(hide_code=True)
def _(np, ts):
    def concentric_shells(nz, ny, nx):
        """Three concentric Gaussian shells; float32 range [0, 1]."""
        z, y, x = np.ogrid[-1 : 1 : nz * 1j, -1 : 1 : ny * 1j, -1 : 1 : nx * 1j]
        r = np.sqrt(x**2 + y**2 + z**2)
        vol = (
            np.exp(-(((r - 0.3) / 0.05) ** 2))
            + np.exp(-(((r - 0.6) / 0.05) ** 2))
            + np.exp(-(((r - 0.9) / 0.05) ** 2))
        )
        return (vol / vol.max()).astype(np.float32)

    def block_average(arr, factor):
        """Downsample by averaging non-overlapping (factor,)*ndim blocks."""
        s = arr.shape
        d = arr.ndim
        slices = tuple(slice(None, (s[i] // factor) * factor) for i in range(d))
        trimmed = arr[slices]
        shape = []
        for i in range(d):
            shape.extend([trimmed.shape[i] // factor, factor])
        return (
            trimmed.reshape(shape)
            .mean(axis=tuple(range(1, 2 * d, 2)))
            .astype(np.float32)
        )

    def write_zarr3(base_path, name, data):
        """Write *data* as a zarr v3 array at *base_path / name*."""
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(base_path / name)},
            "metadata": {
                "shape": list(data.shape),
                "data_type": "float32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [16, 16, 16]},
                },
            },
            "create": True,
            "delete_existing": True,
        }
        ts_store = ts.open(spec).result()
        ts_store[...].write(data).result()

    def make_dataset_info(store, volume):
        """Return an HTML table summarising the multiscale dataset."""
        rows = [
            ("Shape (level 0)", " x ".join(str(s) for s in volume.shape)),
            ("Data type", str(volume.dtype)),
            ("Value range", f"[{volume.min():.3f}, {volume.max():.3f}]"),
            ("Scale levels", str(len(store.scale_names))),
            ("Level names", ", ".join(store.scale_names)),
            ("Level 1 scale", "2x isotropic"),
            ("Level 2 scale", "4x isotropic"),
        ]
        html = ["<table>"]
        html.append("<tr><th>Property</th><th>Value</th></tr>")
        for k, v in rows:
            html.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
        html.append("</table>")
        return "".join(html)

    return (
        block_average,
        concentric_shells,
        make_dataset_info,
        write_zarr3,
    )


@app.cell
def _(Path, block_average, concentric_shells, tempfile, write_zarr3):
    tmpdir = Path(tempfile.mkdtemp())

    volume = concentric_shells(64, 128, 128)
    s1 = block_average(volume, 2)  # (32, 64, 64)
    s2 = block_average(s1, 2)  # (16, 32, 32)

    write_zarr3(tmpdir, "s0", volume)
    write_zarr3(tmpdir, "s1", s1)
    write_zarr3(tmpdir, "s2", s2)

    print(f"s0: {volume.shape}  s1: {s1.shape}  s2: {s2.shape}")
    print(f"value range: [{volume.min():.3f}, {volume.max():.3f}]")
    return tmpdir, volume


@app.cell
def _(
    AffineTransform, MultiscaleZarrDataStore, Viewer, make_dataset_info, tmpdir, volume
):
    store = MultiscaleZarrDataStore(
        zarr_path=str(tmpdir),
        scale_names=["s0", "s1", "s2"],
        level_transforms=[
            AffineTransform.identity(ndim=3),
            AffineTransform.from_scale_and_translation(
                (2.0, 2.0, 2.0), (0.5, 0.5, 0.5)
            ),
            AffineTransform.from_scale_and_translation(
                (4.0, 4.0, 4.0), (1.5, 1.5, 1.5)
            ),
        ],
    )

    viewer = Viewer(axis_labels=("z", "y", "x"), dim="3d", gui="anywidget")

    viewer.add_image_multiscale(
        store,
        appearance={
            "color_map": "viridis",
            "clim": (0.0, 1.0),
            "render_mode": "iso",
            "iso_threshold": 0.45,
            "lod_bias": 1.0,
            "attenuation": 1.0,
        },
        controls={
            "appearance": [
                "color_map",
                "clim",
                "render_mode",
                "iso_threshold",
                "attenuation",
                "lod_bias",
            ],
            "colormap_names": [
                "grays",
                "viridis",
                "plasma",
                "inferno",
                "magma",
                "cividis",
                "turbo",
                "hot",
                "bwr",
                "RdYlBu",
            ],
            "clim_range": (0.0, 1.0),
            "dataset_info": make_dataset_info(store, volume),
        },
    )
    return store, viewer


@app.cell
def _(axis_ranges_from_viewer, build_canvas_widget, viewer):
    axis_ranges = axis_ranges_from_viewer(viewer)

    canvas_view = build_canvas_widget(
        viewer,
        axis_ranges,
        gui="anywidget",
        canvas_size=(400, 400),
    )
    return (canvas_view,)


@app.cell
def _(AppearanceControls, Layout, SceneControls, canvas_view, display, viewer):
    display(
        viewer,
        Layout(
            center=canvas_view,
            left_dock=AppearanceControls(),
            bottom_dock=SceneControls(),
        ),
        fit="ready",
    )
    return


if __name__ == "__main__":
    app.run()
