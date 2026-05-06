"""Tests for multichannel image visuals (model + render layer + controller)."""

from __future__ import annotations

import numpy as np
import pytest

from cellier.v2._state import AxisAlignedSelectionState, DimsState
from cellier.v2.controller import CellierController
from cellier.v2.data.image._image_memory_store import ImageMemoryStore
from cellier.v2.render.visuals._image_memory_multichannel import (
    GFXMultichannelImageMemoryVisual,
)
from cellier.v2.scene.dims import CoordinateSystem
from cellier.v2.visuals._channel_appearance import ChannelAppearance
from cellier.v2.visuals._image_memory import MultichannelImageVisual

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_channel_appearance(**kwargs) -> ChannelAppearance:
    defaults = {"colormap": "viridis", "clim": (0.0, 1.0)}
    defaults.update(kwargs)
    return ChannelAppearance(**defaults)


def _make_4d_store(shape=(3, 2, 16, 16)) -> ImageMemoryStore:
    """Return a float32 in-memory store with shape (Z, C, Y, X)."""
    data = np.random.default_rng(0).random(shape).astype(np.float32)
    return ImageMemoryStore(data=data)


def _make_dims_state_2d(shape, channel_axis=1) -> DimsState:
    """Return a DimsState with displayed_axes=(2, 3) for a 4D (Z,C,Y,X) store."""
    _z_size, _c_size, _y_size, _x_size = shape
    return DimsState(
        axis_labels=("z", "c", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(2, 3),
            slice_indices={0: 0, 1: 0},
        ),
    )


def _make_controller_with_scene() -> tuple[CellierController, object]:
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "c", "y", "x"))
    scene = controller.add_scene(dim="2d", coordinate_system=cs, name="main")
    return controller, scene


# ---------------------------------------------------------------------------
# Model tests (no Qt, no render layer)
# ---------------------------------------------------------------------------


def test_channel_appearance_events_fire_on_clim_change():
    ap = _make_channel_appearance()
    received = []
    ap.events.clim.connect(lambda v: received.append(v))
    ap.clim = (10.0, 200.0)
    assert len(received) == 1
    assert received[0] == (10.0, 200.0)


def test_channel_appearance_events_fire_on_opacity_change():
    ap = _make_channel_appearance()
    received = []
    ap.events.opacity.connect(lambda v: received.append(v))
    ap.opacity = 0.5
    assert len(received) == 1
    assert received[0] == pytest.approx(0.5)


def test_multichannel_image_visual_channels_event_on_replacement():
    channels = {0: _make_channel_appearance(), 1: _make_channel_appearance()}
    visual = MultichannelImageVisual(
        name="test",
        data_store_id="dummy",
        channel_axis=1,
        channels=channels,
    )
    fired = []
    visual.events.channels.connect(lambda v: fired.append(v))
    new_channels = {0: _make_channel_appearance()}
    visual.channels = new_channels
    assert len(fired) == 1
    assert set(fired[0].keys()) == {0}


def test_multichannel_image_visual_channel_axis_is_frozen():
    channels = {0: _make_channel_appearance()}
    visual = MultichannelImageVisual(
        name="test",
        data_store_id="dummy",
        channel_axis=1,
        channels=channels,
    )
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        visual.channel_axis = 2


def test_multichannel_image_visual_construction_defaults():
    channels = {0: _make_channel_appearance(), 1: _make_channel_appearance()}
    visual = MultichannelImageVisual(
        name="test",
        data_store_id="store-xyz",
        channel_axis=1,
        channels=channels,
    )
    assert visual.channel_axis == 1
    assert len(visual.channels) == 2
    assert visual.max_channels_2d == 8
    assert visual.max_channels_3d == 4


# ---------------------------------------------------------------------------
# Render-layer tests (no Qt, no GPU)
# ---------------------------------------------------------------------------


def _make_gfx_visual(channels=None, render_modes=None):
    if channels is None:
        channels = {
            0: _make_channel_appearance(),
            1: _make_channel_appearance(),
        }
    if render_modes is None:
        render_modes = {"2d"}

    store = _make_4d_store(shape=(3, 2, 16, 16))
    visual_model = MultichannelImageVisual(
        name="test",
        data_store_id=str(store.id),
        channel_axis=1,
        channels=channels,
    )
    gfx = GFXMultichannelImageMemoryVisual(
        visual_model=visual_model,
        data_store=store,
        render_modes=render_modes,
    )
    return gfx, visual_model, store


def test_gfx_multichannel_memory_visual_build_slice_request_2d_one_per_channel():
    gfx, _model, store = _make_gfx_visual()
    dims = _make_dims_state_2d(store.shape)

    requests = gfx.build_slice_request_2d(
        camera_pos_world=np.zeros(3, dtype=np.float32),
        viewport_width_px=512.0,
        world_width=16.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=dims,
    )

    assert len(requests) == 2
    channel_indices = {int(r.axis_selections[1]) for r in requests}
    assert channel_indices == {0, 1}


def test_gfx_multichannel_memory_visual_hidden_channel_not_in_requests():
    channels = {
        0: _make_channel_appearance(visible=True),
        1: _make_channel_appearance(visible=False),
    }
    gfx, _model, store = _make_gfx_visual(channels=channels)
    dims = _make_dims_state_2d(store.shape)

    requests = gfx.build_slice_request_2d(
        camera_pos_world=np.zeros(3, dtype=np.float32),
        viewport_width_px=512.0,
        world_width=16.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=dims,
    )

    assert len(requests) == 1
    assert int(requests[0].axis_selections[1]) == 0


def test_gfx_multichannel_memory_visual_on_data_ready_2d_routes_to_correct_slot():
    gfx, _model, store = _make_gfx_visual()
    dims = _make_dims_state_2d(store.shape)

    requests = gfx.build_slice_request_2d(
        camera_pos_world=np.zeros(3, dtype=np.float32),
        viewport_width_px=512.0,
        world_width=16.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=dims,
    )
    assert len(requests) == 2

    batch = []
    for req in requests:
        ch = int(req.axis_selections[1])
        data = np.full((16, 16), float(ch), dtype=np.float32)
        batch.append((req, data))

    gfx.on_data_ready_2d(batch)

    # Each channel's pool node should have non-trivial geometry after delivery.
    for ch_idx in (0, 1):
        slot_idx = gfx._channel_to_slot_2d[ch_idx]
        node = gfx._pool_2d[slot_idx]
        assert node.geometry is not None


def test_gfx_multichannel_memory_visual_channels_replaced_updates_pool():
    gfx, model, _store = _make_gfx_visual()

    # Initially channels 0 and 1 are claimed.
    assert 0 in gfx._channel_to_slot_2d
    assert 1 in gfx._channel_to_slot_2d
    slot_for_1 = gfx._channel_to_slot_2d[1]

    # Replace channels: remove 1, add 2.
    new_channels = {0: _make_channel_appearance(), 2: _make_channel_appearance()}
    model.channels = new_channels

    assert 1 not in gfx._channel_to_slot_2d
    assert 2 in gfx._channel_to_slot_2d
    # Slot previously used by channel 1 should be visible=False.
    assert gfx._pool_2d[slot_for_1].visible is False


# ---------------------------------------------------------------------------
# Controller tests (no Qt event loop needed for basic wiring)
# ---------------------------------------------------------------------------


def test_controller_add_multichannel_image_returns_visual_model():
    controller, scene = _make_controller_with_scene()
    store = _make_4d_store()
    channels = {0: _make_channel_appearance(), 1: _make_channel_appearance()}

    visual = controller.add_multichannel_image(
        data=store,
        scene_id=scene.id,
        channel_axis=1,
        channels=channels,
    )

    assert isinstance(visual, MultichannelImageVisual)
    assert visual in controller._model.scenes[scene.id].visuals
    assert store.id in controller._model.data.stores


def test_controller_add_channel_increments_channels():
    controller, scene = _make_controller_with_scene()
    store = _make_4d_store()
    channels = {0: _make_channel_appearance()}

    visual = controller.add_multichannel_image(
        data=store,
        scene_id=scene.id,
        channel_axis=1,
        channels=channels,
    )
    assert len(visual.channels) == 1

    controller.add_channel(visual.id, 1, _make_channel_appearance())
    assert len(visual.channels) == 2
    assert 1 in visual.channels


def test_controller_add_channel_raises_if_pool_full():
    controller, scene = _make_controller_with_scene()
    store = _make_4d_store(shape=(3, 4, 16, 16))
    channels = {0: _make_channel_appearance(), 1: _make_channel_appearance()}

    visual = controller.add_multichannel_image(
        data=store,
        scene_id=scene.id,
        channel_axis=1,
        channels=channels,
        max_channels_2d=2,
    )

    with pytest.raises(RuntimeError, match="Pool is full"):
        controller.add_channel(visual.id, 2, _make_channel_appearance())


def test_controller_remove_channel_decrements_channels():
    controller, scene = _make_controller_with_scene()
    store = _make_4d_store()
    channels = {0: _make_channel_appearance(), 1: _make_channel_appearance()}

    visual = controller.add_multichannel_image(
        data=store,
        scene_id=scene.id,
        channel_axis=1,
        channels=channels,
    )
    assert len(visual.channels) == 2

    controller.remove_channel(visual.id, 1)
    assert len(visual.channels) == 1
    assert 1 not in visual.channels


def test_controller_add_channel_raises_if_duplicate_index():
    controller, scene = _make_controller_with_scene()
    store = _make_4d_store()
    channels = {0: _make_channel_appearance()}

    visual = controller.add_multichannel_image(
        data=store,
        scene_id=scene.id,
        channel_axis=1,
        channels=channels,
    )

    with pytest.raises(ValueError, match=r"already in visual\.channels"):
        controller.add_channel(visual.id, 0, _make_channel_appearance())


def test_controller_remove_channel_raises_if_not_present():
    controller, scene = _make_controller_with_scene()
    store = _make_4d_store()
    channels = {0: _make_channel_appearance()}

    visual = controller.add_multichannel_image(
        data=store,
        scene_id=scene.id,
        channel_axis=1,
        channels=channels,
    )

    with pytest.raises(KeyError):
        controller.remove_channel(visual.id, 99)
