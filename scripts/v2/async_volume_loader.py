# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pygfx>=0.9.0",
#     "rendercanvas",
#     "tensorstore",
#     "scikit-image",
#     "PySide6",
#     "numpy",
# ]
# ///
"""Prototype for async zarr chunk reading and progressive GPU volume rendering.

Demonstrates an end-to-end async data pipeline:

    tensorstore (async zarr reads, per chunk)
        -> asyncio (PySide6.QtAsyncio event loop bridge)
        -> pygfx Texture (in-place numpy update, main thread)
        -> GPU upload (deferred inside renderer.render())
        -> QRenderWidget (rendercanvas.qt, vsync-driven)

Usage
-----
Generate test data then launch the viewer::

    uv run --script async_volume_loader.py --make-file
    uv run --script async_volume_loader.py

Or point at an existing store::

    uv run --script async_volume_loader.py path/to/blobs.zarr

CLI flags
---------
--make-file
    Write ``multiscale_blobs.zarr`` using tensorstore and exit.  Does not
    launch the viewer.  Requires scikit-image (already listed as a
    dependency in the uv metadata above).

Store layout
------------
::

    multiscale_blobs.zarr/
        level_0/   # 1024³  float32  (~4 GB RAM — excluded from pre-alloc)
        level_1/   #  512³
        level_2/   #  256³
        level_3/   #  128³  <- default (64 chunks of 32³)
        level_4/   #   64³  <- reference isosurface (always loaded at startup)

Architecture notes
------------------
* **Single thread** — PySide6.QtAsyncio installs a Qt-backed asyncio event
  loop.  Qt event processing and asyncio coroutines interleave in the same
  thread via Qt timers — no cross-thread locking is needed for pygfx scene
  mutations.

* **Tensorstore only** — all zarr I/O (metadata, reads, writes) goes through
  tensorstore.  The zarr library is not used.

* **Pre-allocated textures** — shapes and chunk shapes for every level are
  fetched via tensorstore before the window is created.  One zeroed numpy
  array and one ``gfx.Texture`` are allocated per level and reused across
  reloads.  Level 0 (1024³ × 4 bytes ≈ 4 GB) is excluded; change
  ``MIN_LEVEL = 0`` to include it on machines with enough RAM.

* **Physical-space transforms** — pygfx Volume renders each voxel as one
  world unit by default, so a level-n volume occupies ``shape[i]`` world
  units on each axis.  Setting ``volume.local.scale = (2ⁿ, 2ⁿ, 2ⁿ)``
  stretches each voxel to its physical size, making every level occupy the
  same world-space bounding box as level 0.

* **Reference volume** — level 4 (64³) is loaded synchronously at startup
  via ``tensorstore.Future.result()`` and rendered as a
  ``VolumeIsoMaterial`` isosurface (threshold = 0.5).  Its ``gfx.Texture``
  is independent of the pre-allocated dict so it is never zeroed on reload.
  Toggle with the "Show reference" checkbox to verify that blob surfaces
  from the isosurface align with the ray-cast volume at every scale level.

* **Cancellation** — each reload creates one ``asyncio.Task``.  A new reload
  cancels the previous one; ``CancelledError`` propagates naturally through
  the tensorstore ``await`` inside the chunk loop.

* **Texture update** — after each chunk lands, ``texture.data[...]`` is
  updated in-place and ``texture.update_full()`` marks the region dirty.
  The actual GPU ``write_texture`` memcpy is deferred until the next call to
  ``renderer.render()`` inside ``_draw_frame``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import PySide6.QtAsyncio as QtAsyncio
import pygfx as gfx
import tensorstore as ts
from PySide6 import QtCore, QtWidgets
from rendercanvas.qt import QRenderWidget

DEFAULT_ZARR_PATH = Path("multiscale_blobs.zarr")
DEFAULT_LEVEL = 3  # 128³ voxels, 32³ chunks → 64 chunks
REFERENCE_LEVEL = 4  # 64³; loaded synchronously at startup as a reference
MIN_LEVEL = 1  # Level 0 (1024³) needs ~4 GB RAM; set to 0 to include it
MAX_LEVEL = 4
CLIM = (0.0, 1.0)  # intensity window; blobs data is binary float32
ISO_THRESHOLD = 0.5  # isosurface threshold for the reference volume


def physical_scale(level: int) -> tuple[float, float, float]:
    """Return the world-space scale to apply to a ``gfx.Volume`` at this level.

    pygfx Volume renders each voxel as one world unit.  A level-n array has
    ``shape / 2**n`` voxels per axis, each representing ``2**n`` world units.
    Setting ``volume.local.scale = physical_scale(n)`` makes every level
    occupy the same world-space bounding box as level 0.

    Parameters
    ----------
    level : int
        Zarr scale level (0 = full resolution).

    Returns
    -------
    scale : tuple[float, float, float]
        Uniform per-axis scale ``(2**level, 2**level, 2**level)``.
    """
    s = float(2**level)
    return (s, s, s)


def _zarr_spec(zarr_path: Path, level: int) -> dict:
    """Build a tensorstore zarr spec for the given level (read-only open).

    Parameters
    ----------
    zarr_path : Path
        Root path of the multiscale zarr store.
    level : int
        Scale level to address.

    Returns
    -------
    spec : dict
        Tensorstore spec dict suitable for ``ts.open()``.
    """
    return {
        "driver": "zarr",
        "kvstore": {
            "driver": "file",
            "path": str(zarr_path / f"level_{level}"),
        },
    }


async def open_tensorstore_array(zarr_path: Path, level: int) -> ts.TensorStore:
    """Open a zarr level with tensorstore asynchronously.

    Parameters
    ----------
    zarr_path : Path
        Root path of the multiscale zarr store.
    level : int
        Scale level to open (0 = full resolution).

    Returns
    -------
    store : ts.TensorStore
        Open tensorstore handle ready for read operations.
    """
    return await ts.open(_zarr_spec(zarr_path, level))


async def fetch_level_metadata(
    zarr_path: Path, level: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the shape and chunk shape for one zarr level via tensorstore.

    Opens the store just long enough to inspect its spec, then discards the
    handle.  The spec JSON for the zarr driver contains the array shape and
    chunk shape under the ``metadata`` key.

    Parameters
    ----------
    zarr_path : Path
        Root path of the multiscale zarr store.
    level : int
        Scale level to inspect.

    Returns
    -------
    shape : tuple[int, ...]
        Voxel dimensions (Z, Y, X).
    chunk_shape : tuple[int, ...]
        Chunk dimensions (Cz, Cy, Cx).
    """
    store = await open_tensorstore_array(zarr_path, level)
    shape = tuple(int(x) for x in store.domain.shape)
    spec_json = store.spec(minimal_spec=False).to_json()
    chunk_shape = tuple(int(c) for c in spec_json["metadata"]["chunks"])
    return shape, chunk_shape


def iter_chunk_slices(
    shape: tuple[int, ...], chunk_shape: tuple[int, ...]
) -> list[tuple[int, int, int, int, int, int]]:
    """Return all (z0, y0, x0, z1, y1, x1) slices covering the array.

    Parameters
    ----------
    shape : tuple[int, ...]
        Array shape (Z, Y, X).
    chunk_shape : tuple[int, ...]
        Chunk shape (Cz, Cy, Cx).

    Returns
    -------
    slices : list[tuple[int, int, int, int, int, int]]
        One entry per chunk; boundary chunks are clipped to the array shape.
    """
    nz = (shape[0] + chunk_shape[0] - 1) // chunk_shape[0]
    ny = (shape[1] + chunk_shape[1] - 1) // chunk_shape[1]
    nx = (shape[2] + chunk_shape[2] - 1) // chunk_shape[2]
    slices = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                z0, y0, x0 = iz * chunk_shape[0], iy * chunk_shape[1], ix * chunk_shape[2]
                z1 = min(z0 + chunk_shape[0], shape[0])
                y1 = min(y0 + chunk_shape[1], shape[1])
                x1 = min(x0 + chunk_shape[2], shape[2])
                slices.append((z0, y0, x0, z1, y1, x1))
    return slices


async def load_volume_chunked(
    zarr_path: Path,
    level: int,
    shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    texture: gfx.Texture,
    on_progress: callable,
) -> float:
    """Read a zarr level chunk-by-chunk into a pre-allocated pygfx Texture.

    Each chunk is fetched with a separate ``await``, yielding control to the
    Qt/asyncio event loop between reads so the UI and renderer stay live.
    Designed to run as an ``asyncio.Task`` so it can be cancelled at any
    chunk boundary.

    Parameters
    ----------
    zarr_path : Path
        Root path of the multiscale zarr store.
    level : int
        Scale level to load.
    shape : tuple[int, ...]
        Array shape (Z, Y, X); must match ``texture.data.shape``.
    chunk_shape : tuple[int, ...]
        Chunk shape (Cz, Cy, Cx).
    texture : gfx.Texture
        Pre-allocated texture whose ``data`` numpy array is updated in-place
        after each chunk arrives.
    on_progress : callable
        Called after each chunk with ``(chunks_done: int, total_chunks: int)``.

    Returns
    -------
    elapsed : float
        Wall-clock seconds for the full load (including the store-open call).

    Raises
    ------
    asyncio.CancelledError
        Re-raised if the task is cancelled mid-load.
    """
    t_start = time.perf_counter()

    store = await open_tensorstore_array(zarr_path, level)
    slices = iter_chunk_slices(shape, chunk_shape)
    total = len(slices)

    for done, (z0, y0, x0, z1, y1, x1) in enumerate(slices, start=1):
        chunk_np = np.asarray(await store[z0:z1, y0:y1, x0:x1].read())
        texture.data[z0:z1, y0:y1, x0:x1] = chunk_np
        # update_range expects (width, height, depth) = (x, y, z) order,
        # which is the reverse of numpy's (z, y, x) row-major layout.
        texture.update_range(
            offset=(x0, y0, z0),
            size=(x1 - x0, y1 - y0, z1 - z0),
        )
        on_progress(done, total)

    return time.perf_counter() - t_start


def make_test_zarr(zarr_path: Path) -> None:
    """Generate a multiscale binary-blobs zarr store using tensorstore.

    Creates five scale levels (level_0 through level_4) derived from a 1024³
    binary-blobs volume.  Each level is downsampled by 2× relative to the
    previous one using nearest-neighbour interpolation.  Data is written via
    tensorstore (zarr driver, local filesystem kvstore).

    Parameters
    ----------
    zarr_path : Path
        Destination directory for the zarr store.  Created if absent.
        Existing level directories are overwritten.
    """
    from skimage.data import binary_blobs
    from skimage.transform import resize

    base_length = 1024
    chunk_shape = [32, 32, 32]
    n_levels = MAX_LEVEL + 1

    print(f"Generating {base_length}³ binary blobs…")
    image_0 = binary_blobs(
        length=base_length,
        n_dim=3,
        blob_size_fraction=0.2,
        volume_fraction=0.2,
    ).astype(np.float32)

    for level in range(n_levels):
        edge = base_length // (2**level)
        if level == 0:
            data = image_0
        else:
            data = resize(image_0, (edge, edge, edge), order=0).astype(np.float32)

        spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": str(zarr_path / f"level_{level}"),
            },
            "metadata": {
                "dtype": "<f4",
                "shape": list(data.shape),
                "chunks": chunk_shape,
            },
            "create": True,
            "delete_existing": True,
        }
        store = ts.open(spec).result()
        store[...].write(data).result()
        print(f"  Wrote level_{level}: {data.shape}  ({data.nbytes // 2**20} MiB uncompressed)")

    print(f"Done → {zarr_path}")


class VolumeViewerWindow(QtWidgets.QMainWindow):
    """Qt main window with async zarr loading and a pygfx volume renderer.

    Layout
    ------
    Toolbar: Reload | level spinbox | Show reference checkbox | status label
    QRenderWidget: VolumeRayMaterial ray-caster (main, async-loaded)
                 + VolumeIsoMaterial isosurface (reference level 4, toggleable)

    Interaction
    -----------
    * Orbit: click + drag.
    * Zoom: scroll wheel.
    * Reload: cancels any in-flight load and restarts from the current level.
    * Level spinbox: changes scale level and triggers a reload.
    * "Show reference" checkbox: toggles level_4 isosurface.  Blob surfaces
      from the isosurface should overlap the ray-cast blobs at every level
      when scale transforms are correct.

    Parameters
    ----------
    zarr_path : Path
        Root path of the multiscale zarr store.
    level_shapes : dict[int, tuple[int, ...]]
        Pre-fetched voxel shapes keyed by scale level.
    level_chunk_shapes : dict[int, tuple[int, ...]]
        Pre-fetched chunk shapes keyed by scale level.
    level : int
        Initial scale level to display.
    """

    def __init__(
        self,
        zarr_path: Path,
        level_shapes: dict[int, tuple[int, ...]],
        level_chunk_shapes: dict[int, tuple[int, ...]],
        level: int = DEFAULT_LEVEL,
    ) -> None:
        super().__init__()
        self.zarr_path = zarr_path
        self.level_shapes = level_shapes
        self.level_chunk_shapes = level_chunk_shapes
        self.level = level
        self._reload_task: asyncio.Task | None = None

        self.setWindowTitle(f"Async Zarr Volume — level_{level}")
        self.resize(960, 720)

        # Pre-allocate one zeroed numpy array + gfx.Texture per level.
        # These are reused across all reloads; _replace_volume just zeroes them.
        self._vol_data: dict[int, np.ndarray] = {}
        self._textures: dict[int, gfx.Texture] = {}
        for lvl in range(MIN_LEVEL, MAX_LEVEL + 1):
            data = np.zeros(level_shapes[lvl], dtype=np.float32)
            self._vol_data[lvl] = data
            self._textures[lvl] = gfx.Texture(data=data, dim=3)

        self._build_ui()
        self._build_scene()

        self.btn_reload.clicked.connect(self._on_reload_clicked)
        self.spin_level.valueChanged.connect(self._on_level_changed)
        self.chk_ref.stateChanged.connect(self._on_ref_toggled)
        self.canvas.request_draw(self._draw_frame)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        toolbar = QtWidgets.QHBoxLayout()

        self.btn_reload = QtWidgets.QPushButton("⟳  Reload")
        self.btn_reload.setFixedWidth(110)
        self.btn_reload.setToolTip("Cancel any in-flight load and re-read from scratch.")

        self.spin_level = QtWidgets.QSpinBox()
        self.spin_level.setRange(MIN_LEVEL, MAX_LEVEL)
        self.spin_level.setValue(self.level)
        self.spin_level.setPrefix("level ")
        self.spin_level.setFixedWidth(90)
        self.spin_level.setToolTip(
            f"Zarr scale level. {MIN_LEVEL} = ~{2 ** (10 - MIN_LEVEL)}³ voxels, "
            f"{MAX_LEVEL} = coarsest ({2 ** (10 - MAX_LEVEL)}³)."
        )

        self.chk_ref = QtWidgets.QCheckBox(f"Show reference (level_{REFERENCE_LEVEL})")
        self.chk_ref.setChecked(False)
        self.chk_ref.setToolTip(
            f"Toggle the level_{REFERENCE_LEVEL} isosurface reference volume.\n"
            "Blob surfaces should overlap the main ray-cast volume at every\n"
            "scale level when the physical-space transforms are correct."
        )

        self.lbl_status = QtWidgets.QLabel("Initialising…")
        self.lbl_status.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        )

        toolbar.addWidget(self.btn_reload)
        toolbar.addWidget(self.spin_level)
        toolbar.addWidget(self.chk_ref)
        toolbar.addWidget(self.lbl_status, stretch=1)
        root.addLayout(toolbar)

        # update_mode="continuous" → rendercanvas drives _draw_frame every
        # vsync via Qt timer; texture updates appear on the next frame without
        # any explicit canvas.request_draw() call from user code.
        self.canvas = QRenderWidget(central, update_mode="continuous")
        root.addWidget(self.canvas, stretch=1)

    def _build_scene(self) -> None:
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.camera = gfx.PerspectiveCamera(50, 16 / 9)

        self._add_main_volume(self.level)
        self._build_reference_volume()

        self.camera.show_object(self.scene, view_dir=(-1, -1, -1), up=(0, 0, 1))
        self.controller = gfx.OrbitController(
            self.camera, register_events=self.renderer
        )

    def _add_main_volume(self, level: int) -> None:
        """Create a new main Volume from the pre-allocated texture and add it to the scene.

        Parameters
        ----------
        level : int
            Scale level; determines which pre-allocated texture to use and
            what physical-space scale to apply.
        """
        geometry = gfx.Geometry(grid=self._textures[level])
        material = gfx.VolumeRayMaterial(clim=CLIM)
        self.volume = gfx.Volume(geometry, material)
        self.volume.local.scale = physical_scale(level)
        self.scene.add(self.volume)

    def _build_reference_volume(self) -> None:
        """Load level_4 synchronously via tensorstore and add an isosurface reference.

        Uses a separate ``gfx.Texture`` that is never zeroed on reload, so
        the reference isosurface always shows the complete dataset.

        The same ``physical_scale`` transform is applied so the surface aligns
        with the ray-cast volume when the transforms are correct.
        """
        ref_shape = self.level_shapes[REFERENCE_LEVEL]
        ref_data = np.zeros(ref_shape, dtype=np.float32)

        store = ts.open(_zarr_spec(self.zarr_path, REFERENCE_LEVEL)).result()
        ref_data[:] = np.asarray(store.read().result())

        ref_texture = gfx.Texture(data=ref_data, dim=3)
        ref_geometry = gfx.Geometry(grid=ref_texture)
        ref_material = gfx.VolumeIsoMaterial(clim=CLIM, threshold=ISO_THRESHOLD)

        self.vol_ref = gfx.Volume(ref_geometry, ref_material)
        self.vol_ref.local.scale = physical_scale(REFERENCE_LEVEL)
        self.vol_ref.visible = False
        self.scene.add(self.vol_ref)

    def _replace_volume(self, level: int) -> None:
        """Swap the main Volume for one at a different scale level.

        Re-uses the pre-allocated texture for ``level``; zeroes its backing
        array so the volume starts empty while chunks load.  A new
        ``gfx.Volume`` wrapper is created because pygfx does not support
        swapping a Volume's geometry in-place.

        Parameters
        ----------
        level : int
            The new scale level to switch to.
        """
        self.scene.remove(self.volume)
        self._vol_data[level][:] = 0.0
        self._textures[level].update_full()
        self._add_main_volume(level)
        self.camera.show_object(self.scene, view_dir=(-1, -1, -1), up=(0, 0, 1))
        self.setWindowTitle(f"Async Zarr Volume — level_{level}")

    def _draw_frame(self) -> None:
        self.renderer.render(self.scene, self.camera)

    def _schedule_reload(self) -> None:
        """Cancel any running reload task and schedule a fresh one.

        Single entry point for all reload triggers (button, level change,
        initial auto-load).
        """
        if self._reload_task is not None and not self._reload_task.done():
            self._reload_task.cancel()
        self._reload_task = asyncio.ensure_future(self._do_reload())

    def _on_reload_clicked(self) -> None:
        self._schedule_reload()

    def _on_level_changed(self, new_level: int) -> None:
        self.level = new_level
        self._schedule_reload()

    def _on_ref_toggled(self, state: int) -> None:
        self.vol_ref.visible = bool(state)

    async def _do_reload(self) -> None:
        """Async task: reset the texture for the current level then stream chunks.

        Everything before the first ``await`` is synchronous and therefore
        not cancellable, which ensures the scene graph is always consistent.
        The chunk loop is fully cancellable at each tensorstore ``await``.
        """
        self.btn_reload.setEnabled(False)
        self.lbl_status.setText(f"Setting up level_{self.level}…")

        try:
            # Synchronous setup — happens atomically before any await.
            self._replace_volume(self.level)

            self.lbl_status.setText("Opening tensorstore store…")
            elapsed = await load_volume_chunked(
                zarr_path=self.zarr_path,
                level=self.level,
                shape=self.level_shapes[self.level],
                chunk_shape=self.level_chunk_shapes[self.level],
                texture=self._textures[self.level],
                on_progress=self._on_progress,
            )
            self.lbl_status.setText(
                f"level_{self.level} loaded in {elapsed:.2f}s ✓"
            )
        except asyncio.CancelledError:
            self.lbl_status.setText("Load cancelled.")
            raise
        except Exception as exc:
            self.lbl_status.setText(f"Error: {exc}")
            raise
        finally:
            self.btn_reload.setEnabled(True)

    def _on_progress(self, chunks_done: int, total_chunks: int) -> None:
        self.lbl_status.setText(
            f"Loading level_{self.level}…  {chunks_done} / {total_chunks} chunks"
        )


async def async_main(zarr_path: Path) -> None:
    """Top-level coroutine driven by PySide6.QtAsyncio.

    Fetches all level metadata via tensorstore before creating the window so
    that the constructor can pre-allocate textures synchronously.

    Parameters
    ----------
    zarr_path : Path
        Root path of the multiscale zarr store to visualise.
    """
    app = QtWidgets.QApplication.instance()

    print("Fetching level metadata…")
    level_shapes: dict[int, tuple[int, ...]] = {}
    level_chunk_shapes: dict[int, tuple[int, ...]] = {}
    for level in range(MIN_LEVEL, MAX_LEVEL + 1):
        shape, chunk_shape = await fetch_level_metadata(zarr_path, level)
        level_shapes[level] = shape
        level_chunk_shapes[level] = chunk_shape
        print(f"  level_{level}: shape={shape}  chunks={chunk_shape}")

    window = VolumeViewerWindow(
        zarr_path=zarr_path,
        level_shapes=level_shapes,
        level_chunk_shapes=level_chunk_shapes,
        level=DEFAULT_LEVEL,
    )
    window.show()
    window._schedule_reload()

    close_event = asyncio.Event()
    app.aboutToQuit.connect(close_event.set)
    await close_event.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Async zarr volume viewer / test-data generator."
    )
    parser.add_argument(
        "zarr_path",
        nargs="?",
        default=DEFAULT_ZARR_PATH,
        type=Path,
        help="Path to the multiscale zarr store (default: %(default)s).",
    )
    parser.add_argument(
        "--make-file",
        action="store_true",
        help=(
            "Generate the test zarr store at ZARR_PATH using tensorstore "
            "and exit without launching the viewer."
        ),
    )
    args = parser.parse_args()

    if args.make_file:
        make_test_zarr(args.zarr_path)
        sys.exit(0)

    if not args.zarr_path.exists():
        print(f"Error: zarr store not found at '{args.zarr_path}'")
        print("Run with --make-file to generate it:")
        print(f"    uv run --script {Path(__file__).name} --make-file")
        sys.exit(1)

    # Pass only the program name to Qt so argparse flags do not confuse it.
    app = QtWidgets.QApplication([sys.argv[0]])
    QtAsyncio.run(async_main(args.zarr_path), handle_sigint=True)