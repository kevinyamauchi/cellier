"""Render + commit tests for ``GFXMultiscaleLabelVisual`` (Phase 3).

Drives a real multiscale label pyramid through the controller: reslice commits
bricks/tiles to the GPU (the ``on_data_ready`` / material-build path that the
planning-only tests never reach) and the offscreen harness reads back pixels.
Uses the shared fixtures in ``conftest.py`` (``render_scene``, ``reslice``,
``multiscale_labels_store``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from cellier.events._events import (
    AppearanceChangedEvent,
    VisualVisibilityChangedEvent,
)
from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)

if TYPE_CHECKING:
    from cellier.render.visuals._label_multiscale import GFXMultiscaleLabelVisual

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gfx_visual(controller, scene_id, visual_id) -> GFXMultiscaleLabelVisual:
    return controller._render_manager._scenes[scene_id].get_visual(visual_id)


def _opaque_colors(frame: np.ndarray) -> np.ndarray:
    opaque = frame[..., 3] > 0
    return np.unique(frame[opaque][:, :3], axis=0)


# ---------------------------------------------------------------------------
# Construction / node hierarchy
# ---------------------------------------------------------------------------


def test_construction_2d_scene_builds_only_2d_node(controller, multiscale_labels_store):
    """A 2D scene builds ``node_2d`` (Group) and defers ``node_3d``."""
    scene = controller.add_scene(dim="2d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(),
    )
    controller.add_canvas(scene_id=scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert gfx.node_2d is not None
    assert gfx._inner_node_2d is not None
    assert gfx.node_3d is None
    # 2D render owns a 2D block cache, not a 3D one.
    assert gfx._block_cache_2d is not None
    assert gfx._block_cache_3d is None


def test_construction_3d_scene_builds_3d_node(controller, multiscale_labels_store):
    """A 3D scene builds ``node_3d`` and its 3D brick cache."""
    scene = controller.add_scene(dim="3d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(),
    )
    controller.add_canvas(scene_id=scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert gfx.node_3d is not None
    assert gfx._inner_node_3d is not None
    assert gfx._block_cache_3d is not None


# ---------------------------------------------------------------------------
# Rendered output
# ---------------------------------------------------------------------------


async def test_render_2d_shows_labels(
    controller, render_scene, reslice, multiscale_labels_store
):
    """The 2D tile pyramid commits and renders the two labelled blocks."""
    scene = controller.add_scene(dim="2d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(),
        render_config=MultiscaleLabelRenderConfig(block_size=8),
    )
    controller.add_canvas(scene_id=scene.id)

    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id)

    assert np.count_nonzero(frame[..., 3]) > 0
    # Two distinct labels under the random colormap -> at least two colours.
    assert len(_opaque_colors(frame)) >= 2

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert len(gfx._block_cache_2d.tile_manager.tilemap) > 0


async def test_render_3d_commits_bricks_and_draws(
    controller, render_scene, reslice, multiscale_labels_store
):
    """The 3D brick pyramid commits at least one brick and draws opaque pixels."""
    scene = controller.add_scene(dim="3d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(force_level=1),
        render_config=MultiscaleLabelRenderConfig(block_size=8),
    )
    controller.add_canvas(scene_id=scene.id)

    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert len(gfx._block_cache_3d.tile_manager.tilemap) > 0

    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0


# ---------------------------------------------------------------------------
# Direct colormap mode (binds the direct-LUT textures in get_bindings)
# ---------------------------------------------------------------------------


async def test_render_2d_direct_mode_binds_lut(
    controller, render_scene, reslice, multiscale_labels_store
):
    """2D multiscale labels in direct mode bind the direct-LUT textures.

    Covers the ``label_keys_texture is not None`` branch of
    ``LabelBlockShader.get_bindings`` (``_label_multiscale.py``), unreached by
    the random-mode tests.  The store's labels are 3 and 7.
    """
    scene = controller.add_scene(dim="2d", name="scene")
    controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(
            colormap_mode="direct",
            color_dict={
                3: (1.0, 0.0, 0.0, 1.0),
                7: (0.0, 0.0, 1.0, 1.0),
            },
        ),
        render_config=MultiscaleLabelRenderConfig(block_size=8),
    )
    controller.add_canvas(scene_id=scene.id)

    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0


async def test_render_3d_direct_mode_binds_lut(
    controller, render_scene, reslice, multiscale_labels_store
):
    """3D multiscale labels in direct mode bind the direct-LUT textures.

    Covers the direct-mode branch of ``LabelVolumeBrickShader.get_bindings``
    (``_label_multiscale.py``), unreached by the random-mode 3D tests.
    """
    scene = controller.add_scene(dim="3d", name="scene")
    controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(
            colormap_mode="direct",
            color_dict={
                3: (1.0, 0.0, 0.0, 1.0),
                7: (0.0, 0.0, 1.0, 1.0),
            },
            force_level=1,
        ),
        render_config=MultiscaleLabelRenderConfig(block_size=8),
    )
    controller.add_canvas(scene_id=scene.id)

    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0


# ---------------------------------------------------------------------------
# Appearance + visibility updates (drive the render-layer handlers directly)
# ---------------------------------------------------------------------------


async def test_salt_change_reseeds_colormap_buffer(
    controller, reslice, multiscale_labels_store
):
    """Changing ``salt`` writes the new seed into the label params buffer."""
    scene = controller.add_scene(dim="2d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(),
        render_config=MultiscaleLabelRenderConfig(block_size=8),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    gfx.on_appearance_changed(
        AppearanceChangedEvent(
            source_id=uuid4(),
            visual_id=visual.id,
            field_name="salt",
            new_value=42,
            requires_reslice=False,
        )
    )

    assert gfx._salt == 42
    assert int(gfx._label_params_buffer.data["salt"]) == 42


async def test_render_mode_change_updates_3d_material(
    controller, reslice, multiscale_labels_store
):
    """Switching ``render_mode`` pushes onto the live 3D material."""
    scene = controller.add_scene(dim="3d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(force_level=1),
        render_config=MultiscaleLabelRenderConfig(block_size=8),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    gfx.on_appearance_changed(
        AppearanceChangedEvent(
            source_id=uuid4(),
            visual_id=visual.id,
            field_name="render_mode",
            new_value="flat_categorical",
            requires_reslice=False,
        )
    )
    assert gfx.material_3d.render_mode == "flat_categorical"


async def test_opacity_change_applies_to_material(
    controller, reslice, multiscale_labels_store
):
    scene = controller.add_scene(dim="2d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(),
        render_config=MultiscaleLabelRenderConfig(block_size=8),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    gfx.on_appearance_changed(
        AppearanceChangedEvent(
            source_id=uuid4(),
            visual_id=visual.id,
            field_name="opacity",
            new_value=0.25,
            requires_reslice=False,
        )
    )
    assert gfx.material_2d.opacity == 0.25


async def test_visibility_toggle_hides_render(
    controller, render_scene, reslice, multiscale_labels_store
):
    """Hiding the visual empties the rendered frame; re-showing restores it."""
    scene = controller.add_scene(dim="2d", name="scene")
    visual = controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(),
        render_config=MultiscaleLabelRenderConfig(block_size=8),
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
