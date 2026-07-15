"""Defensive paint-fallback bindings for the tile-cache shaders (Phase 4).

``ImageBlockShader`` and ``LabelBlockShader`` unconditionally sample the paint
cache + LUT, so ``get_bindings`` falls back to 1x1 zero textures when the
material was built without paint resources.  The real multiscale visuals always
pass paint textures, so this branch is unreachable through the normal pipeline.
Here we build a real 2D visual, null the paint textures on the live material
*after* reslice but *before* the first frame, then render: pygfx builds the
pipeline lazily on that first draw, so its own ``get_bindings`` call takes the
fallback branch.  Depends on ``render_scene`` (hence ``offscreen_renderer``), so
a machine without an adapter skips.
"""

from __future__ import annotations

import numpy as np

from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
)
from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)


async def test_image_block_paint_fallback_renders(
    controller, render_scene, reslice, multiscale_image_store
):
    """A 2D multiscale image with nulled paint textures still binds + draws.

    Covers the ``paint_cache_texture is None`` / ``paint_lut_texture is None``
    fallbacks in ``ImageBlockShader.get_bindings`` (``_block_image.py``).
    """
    scene = controller.add_scene(dim="2d", name="scene")
    visual = controller.add_image_multiscale(
        data=multiscale_image_store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(color_map="viridis", clim=(0.0, 1.0)),
        render_config=MultiscaleImageRenderConfig(block_size=8),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    material = gfx._inner_node_2d.material
    assert material.paint_cache_texture is not None  # normal path builds them

    # Null before the first draw so pygfx builds the pipeline via the fallback.
    material.paint_cache_texture = None
    material.paint_lut_texture = None

    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0


async def test_label_block_paint_fallback_renders(
    controller, render_scene, reslice, multiscale_labels_store
):
    """A 2D multiscale label with nulled paint textures still binds + draws.

    Covers the ``paint_cache_texture is None`` / ``paint_lut_texture is None``
    fallbacks in ``LabelBlockShader.get_bindings`` (``_label_multiscale.py``).
    """
    scene = controller.add_scene(dim="2d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(),
        render_config=MultiscaleLabelRenderConfig(block_size=8),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    material = gfx._inner_node_2d.material
    assert material.paint_cache_texture is not None

    material.paint_cache_texture = None
    material.paint_lut_texture = None

    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0
