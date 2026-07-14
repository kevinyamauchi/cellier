"""Core (headless) tests for the incoming channel-appearance update path.

Exercises the round-trip added in Phase A: emit a
``ChannelAppearanceUpdateEvent`` on the incoming bus, observe the model mutate
and a ``ChannelAppearanceChangedEvent`` come back out with the right
``source_id``; plus the fan-out helper and the mutator preconditions (§5.3).
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest
from cmap import Colormap
from pydantic import ValidationError

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.events import (
    ChannelAppearanceChangedEvent,
    ChannelAppearanceUpdateEvent,
)
from cellier.scene.dims import CoordinateSystem
from cellier.visuals._channel_appearance import ChannelAppearance


def _make_channel_appearance(**kwargs) -> ChannelAppearance:
    defaults = {"color_map": "viridis", "clim": (0.0, 1.0)}
    defaults.update(kwargs)
    return ChannelAppearance(**defaults)


def _make_4d_store(shape=(3, 2, 16, 16)) -> ImageMemoryStore:
    data = np.random.default_rng(0).random(shape).astype(np.float32)
    return ImageMemoryStore(data=data)


def _make_controller_with_scene() -> tuple[CellierController, object]:
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "c", "y", "x"))
    scene = controller.add_scene(dim="2d", coordinate_system=cs, name="main")
    return controller, scene


def _add_multichannel(controller, scene, channels=None):
    store = _make_4d_store()
    if channels is None:
        channels = {0: _make_channel_appearance(), 1: _make_channel_appearance()}
    return controller.add_multichannel_image(
        data=store,
        scene_id=scene.id,
        channel_axis=1,
        channels=channels,
    )


# ---------------------------------------------------------------------------
# Round-trip: incoming update -> model mutation + outgoing changed event
# ---------------------------------------------------------------------------


def test_incoming_update_mutates_model_and_emits_outgoing():
    controller, scene = _make_controller_with_scene()
    visual = _add_multichannel(controller, scene)

    received: list[ChannelAppearanceChangedEvent] = []
    controller._outgoing_events.subscribe(
        ChannelAppearanceChangedEvent, received.append
    )

    widget_id = uuid4()
    controller.incoming_events.emit(
        ChannelAppearanceUpdateEvent(
            source_id=widget_id,
            visual_id=visual.id,
            channel_index=1,
            field="opacity",
            value=0.25,
        )
    )

    assert visual.channels[1].opacity == pytest.approx(0.25)
    assert len(received) == 1
    assert received[0].source_id == widget_id
    assert received[0].visual_id == visual.id
    assert received[0].channel_index == 1
    assert received[0].field_name == "opacity"
    assert received[0].new_value == pytest.approx(0.25)


def test_outgoing_echo_is_noop_at_matching_subscriber():
    controller, scene = _make_controller_with_scene()
    visual = _add_multichannel(controller, scene)

    widget_id = uuid4()
    applied: list[ChannelAppearanceChangedEvent] = []

    def stub_handler(event: ChannelAppearanceChangedEvent) -> None:
        # Mirror a widget's inbound echo filter.
        if event.source_id == widget_id:
            return
        applied.append(event)

    controller._outgoing_events.subscribe(ChannelAppearanceChangedEvent, stub_handler)

    # Widget-origin edit: the outbound event carries the widget's source_id.
    controller.incoming_events.emit(
        ChannelAppearanceUpdateEvent(
            source_id=widget_id,
            visual_id=visual.id,
            channel_index=0,
            field="opacity",
            value=0.5,
        )
    )
    assert applied == []

    # Programmatic edit (default source_id != widget) is applied.
    controller.update_channel_appearance_field(visual.id, 0, "opacity", 0.75)
    assert len(applied) == 1


# ---------------------------------------------------------------------------
# Preconditions (design §5.3)
# ---------------------------------------------------------------------------


def test_setattr_emits_field_signal():
    appearance = _make_channel_appearance()
    received = []
    appearance.events.opacity.connect(received.append)
    appearance.opacity = 0.5
    assert received == [pytest.approx(0.5)]


def test_noop_assignment_emits_nothing():
    controller, scene = _make_controller_with_scene()
    visual = _add_multichannel(controller, scene)

    received: list[ChannelAppearanceChangedEvent] = []
    controller._outgoing_events.subscribe(
        ChannelAppearanceChangedEvent, received.append
    )

    current = visual.channels[0].opacity
    controller.update_channel_appearance_field(visual.id, 0, "opacity", current)
    assert received == []


def test_clim_list_coerces_to_tuple():
    controller, scene = _make_controller_with_scene()
    visual = _add_multichannel(controller, scene)

    controller.update_channel_appearance_field(visual.id, 0, "clim", [10.0, 200.0])
    assert visual.channels[0].clim == (10.0, 200.0)
    assert isinstance(visual.channels[0].clim, tuple)


def test_color_map_str_coerces_to_colormap():
    controller, scene = _make_controller_with_scene()
    visual = _add_multichannel(controller, scene)

    controller.update_channel_appearance_field(visual.id, 0, "color_map", "magma")
    assert isinstance(visual.channels[0].color_map, Colormap)


def test_malformed_value_raises_validation_error():
    controller, scene = _make_controller_with_scene()
    visual = _add_multichannel(controller, scene)

    with pytest.raises(ValidationError):
        controller.update_channel_appearance_field(
            visual.id, 0, "opacity", "not-a-number"
        )


# ---------------------------------------------------------------------------
# Fan-out helper (design §6.1)
# ---------------------------------------------------------------------------


def test_fan_out_updates_all_visuals():
    controller, scene = _make_controller_with_scene()
    visual0 = _add_multichannel(controller, scene)
    visual1 = _add_multichannel(controller, scene)

    received: list[ChannelAppearanceChangedEvent] = []
    controller._outgoing_events.subscribe(
        ChannelAppearanceChangedEvent, received.append
    )

    controller.update_channel_group_field([visual0.id, visual1.id], 0, "opacity", 0.3)

    assert visual0.channels[0].opacity == pytest.approx(0.3)
    assert visual1.channels[0].opacity == pytest.approx(0.3)
    assert len(received) == 2
    assert {e.visual_id for e in received} == {visual0.id, visual1.id}


def test_single_visual_update_mutates_only_that_visual():
    controller, scene = _make_controller_with_scene()
    visual0 = _add_multichannel(controller, scene)
    visual1 = _add_multichannel(controller, scene)

    baseline = visual0.channels[0].opacity
    controller.update_channel_appearance_field(visual1.id, 0, "opacity", 0.9)

    assert visual1.channels[0].opacity == pytest.approx(0.9)
    assert visual0.channels[0].opacity == pytest.approx(baseline)
