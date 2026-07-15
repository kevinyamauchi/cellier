"""Render + commit tests for ``GFXMultiscaleImageVisual`` (Phase 3).

Exercises the multiscale image commit path -- tile/brick upload, LUT rebuild,
and ``build_material`` -- that the planning-only tests never reach, then reads
back pixels through the offscreen harness.  Shared fixtures come from
``conftest.py`` (``controller``, ``render_scene``, ``reslice``,
``multiscale_image_store``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from cellier.events._events import (
    AppearanceChangedEvent,
    VisualVisibilityChangedEvent,
)
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
)

if TYPE_CHECKING:
    from cellier.render.visuals._image import GFXMultiscaleImageVisual


def _gfx_visual(controller, scene_id, visual_id) -> GFXMultiscaleImageVisual:
    return controller._render_manager._scenes[scene_id].get_visual(visual_id)


def _add(controller, scene_id, store, appearance, block_size=8):
    return controller.add_image_multiscale(
        data=store,
        scene_id=scene_id,
        appearance=appearance,
        render_config=MultiscaleImageRenderConfig(block_size=block_size),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_2d_builds_2d_material(controller, multiscale_image_store):
    """A 2D scene builds ``node_2d`` + a 2D image material and tile cache."""
    scene = controller.add_scene(dim="2d", name="scene")
    visual = _add(
        controller,
        scene.id,
        multiscale_image_store,
        MultiscaleImageAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert gfx.node_2d is not None
    assert gfx.material_2d is not None
    assert gfx.material_3d is None
    assert gfx._block_cache_2d is not None


def test_construction_3d_builds_3d_material(controller, multiscale_image_store):
    """A 3D scene builds ``node_3d`` + a volume material and brick cache."""
    scene = controller.add_scene(dim="3d", name="scene")
    visual = _add(
        controller,
        scene.id,
        multiscale_image_store,
        MultiscaleImageAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert gfx.node_3d is not None
    assert gfx.material_3d is not None
    assert gfx._block_cache_3d is not None


# ---------------------------------------------------------------------------
# Rendered output
# ---------------------------------------------------------------------------


async def test_render_2d_commits_tiles_and_draws(
    controller, render_scene, reslice, multiscale_image_store
):
    """The 2D pyramid commits tiles (LUT + material build) and draws pixels."""
    scene = controller.add_scene(dim="2d", name="scene")
    visual = _add(
        controller,
        scene.id,
        multiscale_image_store,
        MultiscaleImageAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)

    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert len(gfx._block_cache_2d.tile_manager.tilemap) > 0

    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0


async def test_render_3d_mip_commits_bricks_and_draws(
    controller, render_scene, reslice, multiscale_image_store
):
    """The 3D pyramid commits bricks and MIP-renders the bright interior."""
    scene = controller.add_scene(dim="3d", name="scene")
    visual = _add(
        controller,
        scene.id,
        multiscale_image_store,
        MultiscaleImageAppearance(
            color_map="viridis",
            clim=(0.0, 1.0),
            render_mode="mip",
            force_level=0,
        ),
    )
    controller.add_canvas(scene_id=scene.id)

    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert len(gfx._block_cache_3d.tile_manager.tilemap) > 0

    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0


# ---------------------------------------------------------------------------
# Appearance updates (GPU-only handlers) + visibility
# ---------------------------------------------------------------------------


def _appearance_event(visual_id, field, value):
    return AppearanceChangedEvent(
        source_id=uuid4(),
        visual_id=visual_id,
        field_name=field,
        new_value=value,
        requires_reslice=False,
    )


async def test_colormap_and_clim_updates_apply_to_materials(
    controller, reslice, multiscale_image_store
):
    """``color_map`` and ``clim`` changes push onto the 2D material."""
    scene = controller.add_scene(dim="2d", name="scene")
    visual = _add(
        controller,
        scene.id,
        multiscale_image_store,
        MultiscaleImageAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)

    gfx.on_appearance_changed(_appearance_event(visual.id, "color_map", "magma"))
    assert gfx.material_2d.map is not None

    gfx.on_appearance_changed(_appearance_event(visual.id, "clim", (10.0, 200.0)))
    assert tuple(gfx.material_2d.clim) == (10.0, 200.0)

    gfx.on_appearance_changed(_appearance_event(visual.id, "opacity", 0.5))
    assert gfx.material_2d.opacity == 0.5


async def test_render_mode_and_iso_threshold_update_3d_material(
    controller, reslice, multiscale_image_store
):
    """``render_mode`` / ``iso_threshold`` push onto the live volume material."""
    scene = controller.add_scene(dim="3d", name="scene")
    visual = _add(
        controller,
        scene.id,
        multiscale_image_store,
        MultiscaleImageAppearance(
            color_map="viridis",
            clim=(0.0, 1.0),
            render_mode="iso",
            iso_threshold=0.2,
            force_level=0,
        ),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)

    gfx.on_appearance_changed(_appearance_event(visual.id, "iso_threshold", 0.7))
    assert gfx.material_3d.threshold == 0.7

    gfx.on_appearance_changed(_appearance_event(visual.id, "render_mode", "mip"))
    assert gfx.material_3d.render_mode == "mip"


async def test_visibility_toggle_hides_multiscale_image(
    controller, render_scene, reslice, multiscale_image_store
):
    scene = controller.add_scene(dim="2d", name="scene")
    visual = _add(
        controller,
        scene.id,
        multiscale_image_store,
        MultiscaleImageAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    gfx.on_visibility_changed(
        VisualVisibilityChangedEvent(
            source_id=uuid4(), visual_id=visual.id, visible=False
        )
    )
    hidden = render_scene(controller, scene.id)
    assert np.count_nonzero(hidden[..., 3]) == 0

    gfx.on_visibility_changed(
        VisualVisibilityChangedEvent(
            source_id=uuid4(), visual_id=visual.id, visible=True
        )
    )
    shown = render_scene(controller, scene.id)
    assert np.count_nonzero(shown[..., 3]) > 0


def test_volume_material_is_brick_material(controller, multiscale_image_store):
    """The 3D inner node uses the multiscale brick volume material."""
    from cellier.render.visuals._image import MultiscaleVolumeBrickMaterial

    scene = controller.add_scene(dim="3d", name="scene")
    visual = _add(
        controller,
        scene.id,
        multiscale_image_store,
        MultiscaleImageAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)

    gfx_visual = _gfx_visual(controller, scene.id, visual.id)
    assert isinstance(gfx_visual.material_3d, MultiscaleVolumeBrickMaterial)
    # A real colormap map is attached.
    assert gfx_visual.material_3d.map is not None
