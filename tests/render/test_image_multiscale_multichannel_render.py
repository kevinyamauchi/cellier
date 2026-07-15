"""Render + commit tests for ``GFXMultichannelMultiscaleImageVisual`` (Phase 3).

Drives a 2-channel (CZYX) multiscale pyramid through the controller: each
channel occupies a slot, reslice fans requests out per channel, and the slots
composite additively.  Covers channel fan-out, per-channel colormap/clim, and
additive blending output.  Uses the shared harness fixtures (``controller``,
``render_scene``, ``reslice``); the CZYX store + stacked-axis scene are built
locally because they are specific to the multichannel layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pytest
import tensorstore as ts

from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.events._events import ChannelAppearanceChangedEvent
from cellier.scene.dims import AxisAlignedSelection, CoordinateSystem, DimsManager
from cellier.scene.scene import Scene
from cellier.visuals._channel_appearance import ChannelAppearance
from cellier.visuals._image import MultiscaleImageRenderConfig

if TYPE_CHECKING:
    from cellier.render.visuals._image_multiscale_multichannel import (
        GFXMultichannelMultiscaleImageVisual,
    )

# ---------------------------------------------------------------------------
# Fixtures: CZYX multiscale store + stacked-axis scene
# ---------------------------------------------------------------------------


@pytest.fixture
def multichannel_store(tmp_path) -> MultiscaleZarrDataStore:
    """A 2-level (C=2) CZYX pyramid: ch0 bright low-corner, ch1 bright high-corner.

    ``level_transforms`` are full 4D (channel axis scale 1) so the render layer's
    ``expand_dims`` is a no-op and ``level_transforms[0]`` is the identity.
    """
    for name, shape in [("s0", (2, 16, 16, 16)), ("s1", (2, 8, 8, 8))]:
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(tmp_path / name)},
            "metadata": {
                "shape": list(shape),
                "data_type": "float32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {
                        "chunk_shape": [shape[0]] + [s // 2 for s in shape[1:]]
                    },
                },
            },
            "create": True,
            "delete_existing": True,
        }
        store = ts.open(spec).result()
        arr = np.zeros(shape, dtype=np.float32)
        _c, _d, _h, w = shape
        # Split along X so both channels intersect every Z slice: ch0 fills the
        # left half, ch1 the right half.  Any 2D slice then shows both hues.
        arr[0, :, :, : w // 2] = 1.0
        arr[1, :, :, w // 2 :] = 1.0
        store[...].write(arr).result()

    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(tmp_path),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0, 1.0, 1.0), (1.0, 2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0, 0.0), (0.0, 0.5, 0.5, 0.5)],
        name="multichannel_store",
    )


def _make_scene(controller, dim: str) -> Scene:
    """Add a CZYX scene with the channel axis marked as a stacked (composited) axis."""
    cs = CoordinateSystem(name="world", axis_labels=("c", "z", "y", "x"))
    displayed = (1, 2, 3) if dim == "3d" else (2, 3)
    slice_indices = {} if dim == "3d" else {1: 8}
    return controller.add_scene_model(
        Scene(
            name="main",
            dims=DimsManager(
                coordinate_system=cs,
                selection=AxisAlignedSelection(
                    displayed_axes=displayed,
                    slice_indices=slice_indices,
                    stacked_axes=(0,),
                ),
            ),
            render_modes={dim},
        )
    )


def _add_visual(controller, scene, store, channels, dim):
    visual = controller.add_multichannel_image_multiscale(
        data=store,
        scene_id=scene.id,
        channel_axis=0,
        channels=channels,
        render_config=MultiscaleImageRenderConfig(block_size=8),
    )
    controller.add_canvas(scene_id=scene.id, render_modes={dim}, initial_dim=dim)
    return visual


def _gfx_visual(
    controller, scene_id, visual_id
) -> GFXMultichannelMultiscaleImageVisual:
    return controller._render_manager._scenes[scene_id].get_visual(visual_id)


def _has_green(rgb: np.ndarray) -> bool:
    return bool(np.any((rgb[:, 1] > 60) & (rgb[:, 0] < 60) & (rgb[:, 2] < 60)))


def _has_magenta(rgb: np.ndarray) -> bool:
    return bool(np.any((rgb[:, 0] > 60) & (rgb[:, 2] > 60) & (rgb[:, 1] < 60)))


# ---------------------------------------------------------------------------
# Construction / fan-out
# ---------------------------------------------------------------------------


def test_construction_builds_a_slot_per_channel(controller, multichannel_store):
    """Two channels claim two slots under the wrapper group."""
    scene = _make_scene(controller, "2d")
    channels = {
        0: ChannelAppearance(color_map="green", clim=(0.0, 1.0)),
        1: ChannelAppearance(color_map="magenta", clim=(0.0, 1.0)),
    }
    visual = _add_visual(controller, scene, multichannel_store, channels, "2d")

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert len(gfx._slots) >= 2
    assert gfx._group_2d is not None


# ---------------------------------------------------------------------------
# Rendered output (additive blending)
# ---------------------------------------------------------------------------


async def test_render_2d_additive_blend_shows_both_channels(
    controller, render_scene, reslice, multichannel_store
):
    """Both channels commit tiles and composite to distinct coloured regions."""
    scene = _make_scene(controller, "2d")
    channels = {
        0: ChannelAppearance(color_map="green", clim=(0.0, 1.0)),
        1: ChannelAppearance(color_map="magenta", clim=(0.0, 1.0)),
    }
    visual = _add_visual(controller, scene, multichannel_store, channels, "2d")

    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id)

    opaque = frame[..., 3] > 0
    assert np.count_nonzero(opaque) > 0
    # ch0 (green, left half) and ch1 (magenta, right half) both composite in.
    rgb = frame[opaque][:, :3].astype(int)
    assert _has_green(rgb)
    assert _has_magenta(rgb)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    committed = sum(len(s._block_cache_2d.tile_manager.tilemap) for s in gfx._slots)
    assert committed > 0


async def test_render_3d_composites_channels(
    controller, render_scene, reslice, multichannel_store
):
    """The 3D MIP path commits bricks per channel and draws opaque pixels."""
    scene = _make_scene(controller, "3d")
    channels = {
        0: ChannelAppearance(color_map="green", clim=(0.0, 1.0), render_mode_3d="mip"),
        1: ChannelAppearance(
            color_map="magenta", clim=(0.0, 1.0), render_mode_3d="mip"
        ),
    }
    visual = _add_visual(controller, scene, multichannel_store, channels, "3d")

    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    committed = sum(len(s._block_cache_3d.tile_manager.tilemap) for s in gfx._slots)
    assert committed > 0

    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0


# ---------------------------------------------------------------------------
# Per-channel appearance updates
# ---------------------------------------------------------------------------


def _channel_event(visual_id, channel_index, field, value):
    return ChannelAppearanceChangedEvent(
        source_id=uuid4(),
        visual_id=visual_id,
        channel_index=channel_index,
        field_name=field,
        new_value=value,
    )


async def test_channel_clim_and_colormap_update_slot_material(
    controller, reslice, multichannel_store
):
    """Per-channel ``clim`` / ``color_map`` push onto that channel's slot."""
    scene = _make_scene(controller, "2d")
    channels = {
        0: ChannelAppearance(color_map="green", clim=(0.0, 1.0)),
        1: ChannelAppearance(color_map="magenta", clim=(0.0, 1.0)),
    }
    visual = _add_visual(controller, scene, multichannel_store, channels, "2d")
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    slot_idx = gfx._channel_to_slot[1]
    slot = gfx._slots[slot_idx]

    gfx.on_channel_appearance_changed(_channel_event(visual.id, 1, "clim", (5.0, 42.0)))
    assert tuple(slot.material_2d.clim) == (5.0, 42.0)

    gfx.on_channel_appearance_changed(
        _channel_event(visual.id, 1, "color_map", "viridis")
    )
    assert slot.material_2d.map is not None

    gfx.on_channel_appearance_changed(_channel_event(visual.id, 1, "opacity", 0.3))
    assert slot.material_2d.opacity == pytest.approx(0.3)


async def test_channel_visible_toggle_hides_slot(
    controller, render_scene, reslice, multichannel_store
):
    """Hiding one channel drops its slot nodes from the render."""
    scene = _make_scene(controller, "2d")
    channels = {
        0: ChannelAppearance(color_map="green", clim=(0.0, 1.0)),
        1: ChannelAppearance(color_map="magenta", clim=(0.0, 1.0)),
    }
    visual = _add_visual(controller, scene, multichannel_store, channels, "2d")
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    both = render_scene(controller, scene.id)
    both_rgb = both[both[..., 3] > 0][:, :3].astype(int)
    assert _has_green(both_rgb) and _has_magenta(both_rgb)

    # Hide channel 0 (green, left half): its slot node drops out of the render.
    gfx.on_channel_appearance_changed(_channel_event(visual.id, 0, "visible", False))
    slot0 = gfx._slots[gfx._channel_to_slot[0]]
    assert slot0.node_2d.visible is False

    one = render_scene(controller, scene.id)
    one_rgb = one[one[..., 3] > 0][:, :3].astype(int)
    # Green is gone; magenta (channel 1) remains.
    assert not _has_green(one_rgb)
    assert _has_magenta(one_rgb)


async def test_channel_unknown_index_is_ignored(
    controller, reslice, multichannel_store
):
    """A channel-change event for an unmapped index is a safe no-op."""
    scene = _make_scene(controller, "2d")
    channels = {0: ChannelAppearance(color_map="green", clim=(0.0, 1.0))}
    visual = _add_visual(controller, scene, multichannel_store, channels, "2d")
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    # Channel 5 was never added -> handler returns early without raising.
    gfx.on_channel_appearance_changed(_channel_event(visual.id, 5, "opacity", 0.1))
