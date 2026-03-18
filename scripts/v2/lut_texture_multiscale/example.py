"""Interactive multi-LOD volume rendering example.

Run from the phase_1 directory:

    uv run example.py --make-files   # generate the multiscale zarr once
    uv run example.py                # launch the viewer

The viewer starts with an empty cache.  Controls:

- **R / Space** — trigger LOD render pass at current camera position
- **1 / 2 / 3** — force all bricks to that LOD level (for debugging)
- **0** — return to automatic distance-based LOD selection

Each level uses a different constant blob value (0.33 / 0.66 / 1.0)
so the active LOD is visible as a different colour in viridis.
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import sys

import numpy as np
import pygfx as gfx
from rendercanvas.auto import RenderCanvas, loop

from block_volume import make_block_volume

# ── Constants ───────────────────────────────────────────────────────────

ZARR_PATH = pathlib.Path(__file__).parent / "multiscale_blobs.zarr"
BLOCK_SIZE = 32
GPU_BUDGET = 64 * 1024**2  # 64 MB — deliberately small so eviction is visible


# ── Wireframe helper ────────────────────────────────────────────────────

_AABB_EDGE_INDICES = [
    (0, 1), (2, 3), (4, 5), (6, 7),  # vary z
    (0, 2), (1, 3), (4, 6), (5, 7),  # vary y
    (0, 4), (1, 5), (2, 6), (3, 7),  # vary x
]


def make_box_wireframe(
    box_min: np.ndarray,
    box_max: np.ndarray,
    color: str = "white",
) -> gfx.Line:
    """Build a wireframe AABB as ``gfx.Line`` with segment material."""
    corners = np.array(
        list(
            itertools.product(
                [box_min[0], box_max[0]],
                [box_min[1], box_max[1]],
                [box_min[2], box_max[2]],
            )
        ),
        dtype=np.float32,
    )
    positions = np.array(
        [[corners[a], corners[b]] for a, b in _AABB_EDGE_INDICES],
        dtype=np.float32,
    ).reshape(-1, 3)
    geometry = gfx.Geometry(positions=positions)
    material = gfx.LineSegmentMaterial(color=color, thickness=2.0)
    return gfx.Line(geometry, material)


# ── Zarr creation ───────────────────────────────────────────────────────

def make_multiscale_zarr(path: pathlib.Path) -> None:
    """Generate a 3-level multiscale zarr store from skimage blobs.

    Each level uses the same blob pattern (downsampled) with a distinct
    constant value for blobs and zero background.  This makes LOD
    boundaries clearly visible without confounding interpolation
    artifacts.

    - Level 0 (finest, 128³): blobs = 0.33, background = 0
    - Level 1 (64³):          blobs = 0.66, background = 0
    - Level 2 (coarsest, 32³): blobs = 1.0,  background = 0

    Parameters
    ----------
    path : pathlib.Path
        Output directory for the zarr store.
    """
    import zarr
    from skimage.data import binary_blobs
    from skimage.measure import block_reduce
    from skimage.segmentation import clear_border

    print(f"Generating multiscale zarr at {path} ...")

    base = binary_blobs(
        length=128, n_dim=3, blob_size_fraction=0.1, volume_fraction=0.3, rng=42
    ).astype(np.float32)

    # clear blobs touching the border
    # base = clear_border(base).astype(np.float32)

    # Max-pool to preserve blob structure at coarser levels.
    level_0 = base #* 0.33
    level_1 = block_reduce(base, (2, 2, 2), np.max).astype(np.float32) #* 0.66
    level_2 = block_reduce(base, (4, 4, 4), np.max).astype(np.float32) #* 1.0

    # Write to zarr.
    store = zarr.open(str(path), mode="w")
    for i, arr in enumerate([level_0, level_1, level_2]):
        store.create_array(
            f"level_{i}",
            data=arr,
            chunks=(BLOCK_SIZE,) * 3,
        )
        print(
            f"  level_{i}: shape={arr.shape}  "
            f"range=[{arr.min():.2f}, {arr.max():.2f}]"
        )

    print("Done.\n")


def load_multiscale_zarr(path: pathlib.Path) -> list[np.ndarray]:
    """Load all levels from a multiscale zarr store.

    Parameters
    ----------
    path : pathlib.Path
        Path to the zarr store.

    Returns
    -------
    levels : list[np.ndarray]
        ``[finest, ..., coarsest]`` as float32 arrays.
    """
    import zarr

    store = zarr.open(str(path), mode="r")
    levels = []
    i = 0
    while f"level_{i}" in store:
        arr = np.asarray(store[f"level_{i}"], dtype=np.float32)
        print(
            f"  Loaded level_{i}: shape={arr.shape}  "
            f"range=[{arr.min():.2f}, {arr.max():.2f}]"
        )
        levels.append(arr)
        i += 1
    return levels


# ── Viewer ──────────────────────────────────────────────────────────────

def main() -> None:
    """Launch the multi-LOD volume viewer."""
    parser = argparse.ArgumentParser(description="Phase 1 multi-LOD volume viewer")
    parser.add_argument(
        "--make-files",
        action="store_true",
        help="Generate the multiscale zarr and exit.",
    )
    parser.add_argument(
        "--zarr-path",
        type=pathlib.Path,
        default=ZARR_PATH,
        help="Path to the multiscale zarr store.",
    )
    args = parser.parse_args()

    if args.make_files:
        make_multiscale_zarr(args.zarr_path)
        sys.exit(0)

    if not args.zarr_path.exists():
        print(f"Error: zarr store not found at '{args.zarr_path}'")
        print("Run with --make-files first:")
        print("    uv run example.py --make-files")
        sys.exit(1)

    print("Loading multiscale data ...")
    levels = load_multiscale_zarr(args.zarr_path)
    print(f"  {len(levels)} levels loaded.\n")

    vol_obj, state = make_block_volume(
        levels,
        block_size=BLOCK_SIZE,
        gpu_budget_bytes=GPU_BUDGET,
        clim=(0.0, 1.0),
        threshold=0.2,
    )

    print(
        f"Cache: {state.cache_info.grid_side}³ grid = "
        f"{state.cache_info.n_slots} slots "
        f"({state.cache_info.cache_shape[0]}³ voxels, "
        f"overlap={state.cache_info.overlap}, "
        f"padded_brick={state.cache_info.padded_block_size}³)"
    )
    print(f"LUT:   {state.base_layout.grid_dims} grid")
    print(f"Levels: {state.n_levels}\n")

    # ── Canvas + scene ──────────────────────────────────────────────

    canvas = RenderCanvas(
        size=(900, 700),
        title="LUT Volume Renderer — Phase 1 (multi-LOD, iso)",
    )
    renderer = gfx.renderers.WgpuRenderer(canvas)

    scene = gfx.Scene()
    scene.add(vol_obj)

    # Wireframe AABB showing the volume data bounds.
    # pygfx volume positions go from -0.5 to size-0.5 in (x=W, y=H, z=D) order.
    # Expand by 1 voxel so the wireframe sits outside the volume's depth
    # buffer and is always visible.
    d, h, w = state.base_layout.volume_shape
    pad = 1.0
    aabb_min = np.array([-0.5 - pad, -0.5 - pad, -0.5 - pad], dtype=np.float32)
    aabb_max = np.array([w - 0.5 + pad, h - 0.5 + pad, d - 0.5 + pad], dtype=np.float32)
    wireframe = make_box_wireframe(aabb_min, aabb_max, color="white")
    scene.add(wireframe)

    camera = gfx.PerspectiveCamera(70, 16 / 9)
    camera.show_object(scene, view_dir=(-1, -1, -1), up=(0, 0, 1))
    controller = gfx.OrbitController(camera, register_events=renderer)

    # ── Render trigger (press R or Space) ──────────────────────────

    render_triggered = [False]
    current_force_level = [None]  # None = normal LOD, 1/2/3 = forced

    def on_key(event):
        if event.key in ("r", " "):
            render_triggered[0] = True
        elif event.key == "0":
            current_force_level[0] = None
            state.tile_manager.clear()
            render_triggered[0] = True
            print(">> LOD mode: auto (distance-based)")
        elif event.key in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
            level = int(event.key)
            if level <= state.n_levels:
                current_force_level[0] = level
                state.tile_manager.clear()
                render_triggered[0] = True
                print(f">> LOD mode: force_level={level}")
            else:
                print(f">> Level {level} out of range (max {state.n_levels})")

    renderer.add_event_handler(on_key, "key_down")

    print("Controls:")
    print("  Orbit with mouse")
    print("  R / Space  — trigger LOD render pass")
    print("  0          — auto LOD (distance-based)")
    for i in range(1, state.n_levels + 1):
        print(f"  {i}          — force all bricks to level {i}")
    print()

    # ── Animate loop ────────────────────────────────────────────────

    def animate():
        if render_triggered[0]:
            render_triggered[0] = False

            cam_pos = np.array(camera.world.position, dtype=np.float64)

            stats = state.update(cam_pos, force_level=current_force_level[0])

            print(f"--- Frame {state.frame_number} done ---\n")

        renderer.render(scene, camera)
        canvas.request_draw()

    canvas.request_draw(animate)
    loop.run()


if __name__ == "__main__":
    main()
