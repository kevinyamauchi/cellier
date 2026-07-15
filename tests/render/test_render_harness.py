"""Smoke tests for the Phase 2 offscreen render harness (conftest fixtures).

These verify the shared harness itself -- an offscreen frame can be drawn and
read back as pixels -- so the per-visual Phase 3 tests can rely on it.
"""

from __future__ import annotations

import numpy as np

from cellier.visuals._image_memory import InMemoryImageAppearance


async def test_render_scene_returns_rgba_frame(
    controller, render_scene, drive_reslice, gradient_image
):
    """A gradient image renders to a non-empty RGBA frame of the right size."""
    scene = controller.add_scene(dim="2d", name="scene")
    controller.add_image(
        data=gradient_image,
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)

    controller.reslice_all()
    await drive_reslice(controller)

    frame = render_scene(controller, scene.id, size=(96, 96))

    assert frame.shape == (96, 96, 4)
    assert frame.dtype == np.uint8
    # Something opaque was drawn.
    assert np.count_nonzero(frame[..., 3]) > 0


async def test_render_scene_reflects_gradient_orientation(
    controller, render_scene, drive_reslice, gradient_image
):
    """The 0->1 gradient along X maps to viridis dark(left)->bright(right)."""
    scene = controller.add_scene(dim="2d", name="scene")
    controller.add_image(
        data=gradient_image,
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)

    controller.reslice_all()
    await drive_reslice(controller)

    frame = render_scene(controller, scene.id, size=(128, 128))

    opaque = frame[..., 3] > 0
    xs = np.where(opaque.any(axis=0))[0]
    ys = np.where(opaque.any(axis=1))[0]
    row = (ys.min() + ys.max()) // 2
    left = frame[row, xs.min() + 2, :3].astype(int)
    right = frame[row, xs.max() - 2, :3].astype(int)

    # Viridis low value is dark blue/purple, high value is bright yellow-green:
    # the green+red channels climb strongly from the low to the high end.
    assert right.sum() > left.sum()
    assert right[1] > left[1]  # green channel brighter at the high (X) end


async def test_render_scene_background_is_transparent(
    controller, render_scene, drive_reslice, gradient_image
):
    """The frame corners (outside the centred image) are fully transparent.

    ``fit_camera`` frames the image with headroom, so the extreme corners fall
    on empty background -- a stable check that the harness isn't filling the
    whole framebuffer with an opaque clear colour.
    """
    scene = controller.add_scene(dim="2d", name="scene")
    controller.add_image(
        data=gradient_image,
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)

    controller.reslice_all()
    await drive_reslice(controller)

    frame = render_scene(controller, scene.id, size=(128, 128))

    # Image is centred and does not reach the corners.
    assert frame[0, 0, 3] == 0
    assert frame[-1, -1, 3] == 0
    # But the interior has opaque content.
    assert frame[64, 64, 3] > 0
