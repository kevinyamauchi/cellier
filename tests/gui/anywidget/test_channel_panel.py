"""Tests for the anywidget ``ChannelPanel`` composite channel-controls widget."""

from __future__ import annotations

from uuid import uuid4

import pytest
from cmap import Colormap

pytest.importorskip("anywidget")

from cellier.events import ChannelAppearanceChangedEvent  # noqa: E402
from cellier.gui.anywidget import ChannelPanel  # noqa: E402
from cellier.visuals._channel_appearance import ChannelAppearance  # noqa: E402


def _make_channel_appearance(**kwargs) -> ChannelAppearance:
    defaults = {"color_map": "viridis", "clim": (0.0, 1.0)}
    defaults.update(kwargs)
    return ChannelAppearance(**defaults)


def _make_panel(visual_ids=None, channels=None):
    if channels is None:
        channels = {0: _make_channel_appearance(), 1: _make_channel_appearance()}
    if visual_ids is None:
        visual_ids = [uuid4()]
    return ChannelPanel(visual_ids, channels), visual_ids, channels


def test_traits_created_per_channel_field():
    panel, _vids, _channels = _make_panel()
    for i in (0, 1):
        for field in ("visible", "color_map", "clim", "opacity"):
            assert panel.has_trait(f"ch{i}_{field}")
    assert panel.channel_count == 2
    # Seeded from a cmap.Colormap; stored as its canonical name string.
    assert isinstance(panel.ch0_color_map, str)
    assert "viridis" in panel.ch0_color_map


def test_trait_edit_emits_update_event_per_visual():
    vids = [uuid4(), uuid4()]
    panel, _vids, _channels = _make_panel(visual_ids=vids)
    emitted = []
    panel.changed.connect(emitted.append)

    panel.ch1_opacity = 0.25

    assert len(emitted) == len(vids)
    for event, vid in zip(emitted, vids):
        assert event.source_id == panel._id
        assert event.visual_id == vid
        assert event.channel_index == 1
        assert event.field == "opacity"
        assert event.value == pytest.approx(0.25)


def test_clim_edit_emits_tuple():
    panel, _vids, _channels = _make_panel()
    emitted = []
    panel.changed.connect(emitted.append)

    panel.ch0_clim = [10.0, 200.0]

    assert len(emitted) == 1
    assert emitted[0].field == "clim"
    assert emitted[0].value == (10.0, 200.0)
    assert isinstance(emitted[0].value, tuple)


def test_inbound_change_updates_trait_without_reemit():
    vid = uuid4()
    panel, _vids, _channels = _make_panel(visual_ids=[vid])
    emitted = []
    panel.changed.connect(emitted.append)

    panel._on_changed(
        ChannelAppearanceChangedEvent(
            source_id=uuid4(),  # not the panel -> applied
            visual_id=vid,
            channel_index=0,
            field_name="opacity",
            new_value=0.4,
        )
    )

    assert panel.ch0_opacity == pytest.approx(0.4)
    assert emitted == []  # applied under _applying, no echo


def test_inbound_echo_filtered_by_source_id():
    vid = uuid4()
    panel, _vids, _channels = _make_panel(visual_ids=[vid])

    panel._on_changed(
        ChannelAppearanceChangedEvent(
            source_id=panel._id,  # our own echo -> ignored
            visual_id=vid,
            channel_index=0,
            field_name="opacity",
            new_value=0.9,
        )
    )

    assert panel.ch0_opacity == pytest.approx(1.0)  # unchanged default


def test_inbound_colormap_object_becomes_name_string():
    vid = uuid4()
    panel, _vids, _channels = _make_panel(visual_ids=[vid])

    panel._on_changed(
        ChannelAppearanceChangedEvent(
            source_id=uuid4(),
            visual_id=vid,
            channel_index=1,
            field_name="color_map",
            new_value=Colormap("magma"),
        )
    )

    assert isinstance(panel.ch1_color_map, str)
    assert "magma" in panel.ch1_color_map


def test_inbound_clim_object_normalized_to_float_list():
    vid = uuid4()
    panel, _vids, _channels = _make_panel(visual_ids=[vid])

    panel._on_changed(
        ChannelAppearanceChangedEvent(
            source_id=uuid4(),
            visual_id=vid,
            channel_index=0,
            field_name="clim",
            new_value=(5, 9),
        )
    )

    assert panel.ch0_clim == [5.0, 9.0]


def test_custom_fields_only_create_those_traits():
    channels = {0: _make_channel_appearance()}
    panel = ChannelPanel([uuid4()], channels, fields=["visible", "opacity"])
    assert panel.has_trait("ch0_visible")
    assert panel.has_trait("ch0_opacity")
    assert not panel.has_trait("ch0_color_map")
    assert not panel.has_trait("ch0_clim")
