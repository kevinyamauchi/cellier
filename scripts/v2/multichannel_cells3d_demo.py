"""multichannel_cells3d_demo.py — 3D/2D viewer for the cells3d fluorescence volume.

Two-channel MIP volume viewer: cell membranes (green) + nuclei (magenta).
Supports 3D orbit view and 2D z-slice view with per-channel appearance controls.

Usage
-----
Create the example zarr (once)::

    uv run scripts/v2/multichannel_cells3d_demo.py --make-example

Launch the viewer::

    uv run scripts/v2/multichannel_cells3d_demo.py
    uv run scripts/v2/multichannel_cells3d_demo.py --zarr-file-path /path/to/cells3d.ome.zarr

Controls
--------
- **Toggle 2D / 3D** — switch between full-volume MIP and a Z-slice view.
- **Z slice** slider — select the displayed Z plane (2D mode only).
- Per-channel group boxes — colormap, contrast limits, opacity, visibility.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Data creation
# ---------------------------------------------------------------------------


def make_cells3d_zarr(output_path: str) -> None:
    """Write a 3-level OME-Zarr v0.5 store from skimage.data.cells3d().

    Shape at each level (c, z, y, x) — channel-first per OME-Zarr axis ordering rules:
      Level 0: (2,  60, 256, 256)
      Level 1: (2,  60, 128, 128)
      Level 2: (2,  60,  64,  64)
    """
    import zarr
    from skimage.data import cells3d
    from skimage.measure import block_reduce

    print("Loading skimage.data.cells3d() …")
    # skimage returns (z, c, y, x); transpose to (c, z, y, x) so that channel
    # precedes all spatial axes, as required by the OME-Zarr v0.5 axis ordering rule.
    data = np.transpose(cells3d(), (1, 0, 2, 3)).astype(np.uint16)

    levels = [data]
    for _ in range(2):
        prev = levels[-1]
        down = block_reduce(prev, block_size=(1, 1, 2, 2), func=np.mean).astype(
            np.uint16
        )
        levels.append(down)

    store = zarr.open_group(output_path, mode="w")
    for idx, arr in enumerate(levels):
        store.create_array(
            str(idx),
            data=arr,
            chunks=(1, 1, 64, 64),
            overwrite=True,
        )
        print(f"  level {idx}: shape={arr.shape}")

    yx_scales = [0.26 * (2.0**i) for i in range(3)]
    store.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "path": str(i),
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": [1.0, 0.29, yx_scales[i], yx_scales[i]],
                            }
                        ],
                    }
                    for i in range(3)
                ],
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0]}
                ],
            }
        ],
    }
    print(f"Written to {output_path}")


# ---------------------------------------------------------------------------
# Qt viewer class
# ---------------------------------------------------------------------------


class MultichannelCells3dViewer:
    """Main window with canvas + per-channel controls + 2D/3D toggle."""

    def __init__(
        self,
        controller,
        scene,
        visual_model,
        canvas_widget,
        z_max: int,
        ch0_appearance,
        ch1_appearance,
    ):
        from PySide6 import QtCore, QtWidgets

        self._controller = controller
        self._scene = scene
        self._is_3d = True

        self._window = QtWidgets.QMainWindow()
        self._window.setWindowTitle("cells3d — multichannel viewer")
        self._window.resize(1100, 700)

        central = QtWidgets.QWidget()
        self._window.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        root.addWidget(canvas_widget, stretch=1)

        # ── side panel ────────────────────────────────────────────────
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(240)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        root.addWidget(panel)

        # ── 2D/3D toggle ──────────────────────────────────────────────
        self._toggle_btn = QtWidgets.QPushButton("Toggle 2D / 3D")
        self._toggle_btn.clicked.connect(self._on_toggle_clicked)
        panel_layout.addWidget(self._toggle_btn)

        # ── Z slice (hidden in 3D) ─────────────────────────────────────
        z_box = QtWidgets.QGroupBox("Z slice")
        z_layout = QtWidgets.QVBoxLayout(z_box)
        self._z_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._z_slider.setRange(0, z_max)
        self._z_slider.setValue(z_max // 2)
        self._z_slider.valueChanged.connect(self._on_z_changed)
        z_layout.addWidget(self._z_slider)
        z_box.setVisible(False)
        panel_layout.addWidget(z_box)
        self._z_group = z_box

        # ── per-channel groups ─────────────────────────────────────────
        panel_layout.addWidget(
            self._build_channel_group(
                "Channel 0 — Membranes", ch0_appearance, (0, 65535)
            )
        )
        panel_layout.addWidget(
            self._build_channel_group("Channel 1 — Nuclei", ch1_appearance, (0, 65535))
        )
        panel_layout.addStretch()

    # ------------------------------------------------------------------

    def _build_channel_group(self, label: str, ch, clim_range: tuple[float, float]):
        from PySide6 import QtWidgets
        from PySide6.QtCore import Qt
        from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider
        from superqt.cmap import QColormapComboBox

        group = QtWidgets.QGroupBox(label)
        layout = QtWidgets.QVBoxLayout(group)

        # visibility
        vis_cb = QtWidgets.QCheckBox("Visible")
        vis_cb.setChecked(ch.visible)
        vis_cb.stateChanged.connect(
            lambda state, _ch=ch: setattr(_ch, "visible", bool(state))
        )
        ch.events.visible.connect(
            lambda v, _cb=vis_cb: (
                _cb.blockSignals(True),
                _cb.setChecked(v),
                _cb.blockSignals(False),
            )
        )
        layout.addWidget(vis_cb)

        # colormap
        combo = QColormapComboBox()
        combo.setCurrentColormap(ch.colormap)
        combo.currentColormapChanged.connect(
            lambda cmap_obj, _ch=ch: setattr(_ch, "colormap", cmap_obj)
        )
        ch.events.colormap.connect(lambda v, _combo=combo: _combo.setCurrentColormap(v))
        layout.addWidget(combo)

        # clim
        clim_slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        clim_slider.setDecimals(0)
        clim_slider.setRange(*clim_range)
        clim_slider.setValue(ch.clim)
        clim_slider.valueChanged.connect(
            lambda v, _ch=ch: setattr(_ch, "clim", tuple(v))
        )
        ch.events.clim.connect(
            lambda v, _s=clim_slider: (
                _s.blockSignals(True),
                _s.setValue(v),
                _s.blockSignals(False),
            )
        )
        layout.addWidget(clim_slider)

        # opacity
        opacity_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        opacity_slider.setRange(0.0, 1.0)
        opacity_slider.setSingleStep(0.05)
        opacity_slider.setValue(ch.opacity)
        opacity_slider.valueChanged.connect(
            lambda v, _ch=ch: setattr(_ch, "opacity", v)
        )
        ch.events.opacity.connect(
            lambda v, _s=opacity_slider: (
                _s.blockSignals(True),
                _s.setValue(v),
                _s.blockSignals(False),
            )
        )
        layout.addWidget(opacity_slider)

        return group

    def _on_toggle_clicked(self) -> None:
        from cellier.v2.events import DimsUpdateEvent

        coordinator = self._controller._render_manager._slice_coordinator
        coordinator.cancel_scene(self._scene.id)

        if self._is_3d:
            z = self._z_slider.value()
            self._controller.incoming_events.emit(
                DimsUpdateEvent(
                    source_id=self._controller._id,
                    scene_id=self._scene.id,
                    slice_indices={1: z},
                    displayed_axes=(2, 3),
                )
            )
            self._z_group.setVisible(True)
        else:
            self._controller.incoming_events.emit(
                DimsUpdateEvent(
                    source_id=self._controller._id,
                    scene_id=self._scene.id,
                    slice_indices={},
                    displayed_axes=(1, 2, 3),
                )
            )
            self._z_group.setVisible(False)

        self._is_3d = not self._is_3d

    def _on_z_changed(self, z: int) -> None:
        from cellier.v2.events import DimsUpdateEvent

        self._controller.incoming_events.emit(
            DimsUpdateEvent(
                source_id=self._controller._id,
                scene_id=self._scene.id,
                slice_indices={1: z},
                displayed_axes=(2, 3),
            )
        )

    @property
    def window(self):
        return self._window


# ---------------------------------------------------------------------------
# async entry point
# ---------------------------------------------------------------------------


async def async_main(zarr_path: str) -> None:
    from PySide6 import QtWidgets

    from cellier.v2.controller import CellierController
    from cellier.v2.data.image import OMEZarrImageDataStore
    from cellier.v2.scene.dims import (
        AxisAlignedSelection,
        CoordinateSystem,
        DimsManager,
    )
    from cellier.v2.scene.scene import Scene
    from cellier.v2.visuals._channel_appearance import ChannelAppearance

    # resolve local paths to file:// URI
    if "://" not in zarr_path:
        zarr_uri = f"file://{Path(zarr_path).resolve()}"
    else:
        zarr_uri = zarr_path

    print(f"Opening {zarr_uri} …")
    store = OMEZarrImageDataStore.from_path(zarr_uri)
    print(f"  levels: {store.n_levels}")
    for i, shape in enumerate(store.level_shapes):
        print(f"  level {i}: {shape}")

    controller = CellierController()
    controller.add_data_store(store)

    # Data is (c, z, y, x); channel_axis=0. displayed_axes=(1,2,3) covers the
    # three spatial axes while the channel axis (0) is handled via fill=.
    cs = CoordinateSystem(name="world", axis_labels=("c", "z", "y", "x"))
    scene = controller.add_scene_model(
        Scene(
            name="main",
            dims=DimsManager(
                coordinate_system=cs,
                selection=AxisAlignedSelection(
                    displayed_axes=(1, 2, 3),
                    slice_indices={0: 0},
                ),
            ),
            render_modes={"2d", "3d"},
            lighting="none",
        )
    )

    ch0 = ChannelAppearance(colormap="green", clim=(0, 40000), render_mode_3d="mip")
    ch1 = ChannelAppearance(colormap="magenta", clim=(0, 65535), render_mode_3d="mip")

    visual_model = controller.add_multichannel_image_multiscale(
        data=store,
        scene_id=scene.id,
        channel_axis=0,
        channels={0: ch0, 1: ch1},
        name="cells3d",
    )

    # Cells3d in voxel space: z∈[0,59], y/x∈[0,255]. Max extent ~255 voxels.
    canvas_widget = controller.add_canvas(
        scene.id,
        render_modes={"2d", "3d"},
        initial_dim="3d",
        depth_range_3d=(0.025, 2550.0),
    )

    z_max = store.level_shapes[0][1] - 1  # axis 0 is c, axis 1 is z
    viewer = MultichannelCells3dViewer(
        controller=controller,
        scene=scene,
        visual_model=visual_model,
        canvas_widget=canvas_widget,
        z_max=z_max,
        ch0_appearance=ch0,
        ch1_appearance=ch1,
    )
    viewer.window.show()

    controller.fit_camera(scene.id)
    controller.reslice_scene(scene.id)

    # Work around a PySide6 QtAsyncio bug where default_exception_handler
    # unconditionally accesses context['task'], but non-task callbacks
    # (call_soon/call_later) don't include that key.  We still print the
    # exception so real bugs remain visible.
    import traceback as _tb

    _loop = asyncio.get_event_loop()

    def _exception_handler(ctx):
        if "task" not in ctx:
            exc = ctx.get("exception")
            # CancelledError on a GatheringFuture is expected when in-flight
            # slice requests are cancelled (e.g. z-slider drag). Not a bug.
            if isinstance(exc, asyncio.CancelledError):
                return
            msg = ctx.get("message", "(no message)")
            if exc is not None:
                print(f"asyncio exception (no task): {msg}")
                _tb.print_exception(type(exc), exc, exc.__traceback__)
        else:
            _loop.default_exception_handler(ctx)

    _loop.set_exception_handler(_exception_handler)

    app = QtWidgets.QApplication.instance()
    close_event = asyncio.Event()
    app.aboutToQuit.connect(close_event.set)
    await close_event.wait()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--make-example",
        action="store_true",
        help="Create cells3d.ome.zarr next to this script and exit.",
    )
    p.add_argument(
        "--zarr-file-path",
        default=None,
        metavar="PATH",
        help="Path or URI to an existing OME-Zarr store.",
    )
    args = p.parse_args()

    default_path = Path(__file__).parent / "cells3d.ome.zarr"

    if args.make_example:
        make_cells3d_zarr(args.zarr_file_path or str(default_path))
        return

    zarr_path = args.zarr_file_path or str(default_path)
    if (
        not zarr_path.startswith(("file://", "s3://", "gs://", "https://"))
        and not Path(zarr_path).exists()
    ):
        print(
            f"Error: {zarr_path} not found. Run with --make-example first.",
            file=sys.stderr,
        )
        sys.exit(1)

    import PySide6.QtAsyncio as QtAsyncio
    from PySide6.QtWidgets import QApplication

    app = QApplication([sys.argv[0]])
    QtAsyncio.run(async_main(zarr_path), handle_sigint=True)


if __name__ == "__main__":
    main()
