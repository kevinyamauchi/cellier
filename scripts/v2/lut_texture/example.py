"""Interactive LUT-based volume rendering example.

Run from the phase_0 directory with:

    uv run example.py
"""

import numpy as np
import pygfx as gfx
from rendercanvas.auto import RenderCanvas, loop

from block_volume import make_block_volume


def make_demo_volume(
    shape: tuple[int, int, int] = (128, 128, 128),
    seed: int = 0,
) -> np.ndarray:
    """Create a demo volume with smooth blobs for visualisation.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume dimensions (D, H, W).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    volume : np.ndarray
        Float32 array with values in [0, 1].
    """
    rng = np.random.default_rng(seed)
    d, h, w = shape

    zz, yy, xx = np.mgrid[:d, :h, :w].astype(np.float32)
    zz /= d
    yy /= h
    xx /= w

    vol = np.zeros(shape, dtype=np.float32)

    n_blobs = 12
    for _ in range(n_blobs):
        cx, cy, cz = rng.random(3)
        sigma = 0.05 + rng.random() * 0.1
        intensity = 0.3 + rng.random() * 0.7
        blob = np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
            / (2.0 * sigma**2)
        )
        vol += intensity * blob

    vol /= vol.max()
    return vol


def main() -> None:
    """Run the interactive volume rendering demo."""
    volume = make_demo_volume()

    vol_obj = make_block_volume(
        volume,
        block_size=32,
        clim=(0.0, 1.0),
    )

    canvas = RenderCanvas(
        size=(800, 600),
        title="LUT Volume Renderer — Phase 0",
    )
    renderer = gfx.renderers.WgpuRenderer(canvas)
    scene = gfx.Scene()
    scene.add(vol_obj)

    camera = gfx.PerspectiveCamera(70, 16 / 9)
    camera.show_object(scene, view_dir=(-1, -1, -1), up=(0, 0, 1))
    controller = gfx.OrbitController(camera, register_events=renderer)

    def animate():
        renderer.render(scene, camera)
        canvas.request_draw()

    canvas.request_draw(animate)
    loop.run()


if __name__ == "__main__":
    main()
