"""Tests for the anywidget ``AnywidgetVolumeRenderControls`` widget."""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("anywidget")

from cellier.events import AppearanceChangedEvent  # noqa: E402
from cellier.gui.anywidget.visuals._image import (  # noqa: E402
    AnywidgetVolumeRenderControls,
)


def _make_widget(**kwargs):
    visual_id = uuid4()
    return AnywidgetVolumeRenderControls(visual_id, **kwargs), visual_id


def test_instantiate_smoke():
    widget, _visual_id = _make_widget(
        initial_render_mode="iso",
        initial_threshold=0.4,
        initial_attenuation=2.0,
    )
    assert widget.render_mode == "iso"
    assert widget.iso_threshold == pytest.approx(0.4)
    assert widget.attenuation == pytest.approx(2.0)


def test_instantiate_defaults():
    widget, _visual_id = _make_widget()
    assert widget.render_mode == "mip"
    assert widget.iso_threshold == pytest.approx(0.2)
    assert widget.attenuation == pytest.approx(1.0)


@pytest.mark.parametrize(
    "field, value",
    [("render_mode", "attenuated_mip"), ("iso_threshold", 0.7), ("attenuation", 3.5)],
)
def test_trait_edit_emits_update_event(field, value):
    widget, visual_id = _make_widget()
    emitted = []
    widget.changed.connect(emitted.append)

    setattr(widget, field, value)

    assert len(emitted) == 1
    event = emitted[0]
    assert event.source_id == widget._id
    assert event.visual_id == visual_id
    assert event.field == field
    if isinstance(value, float):
        assert event.value == pytest.approx(value)
    else:
        assert event.value == value


def test_inbound_change_updates_trait_without_reemit():
    widget, visual_id = _make_widget()
    emitted = []
    widget.changed.connect(emitted.append)

    widget._on_appearance_changed(
        AppearanceChangedEvent(
            source_id=uuid4(),  # not the widget -> applied
            visual_id=visual_id,
            field_name="iso_threshold",
            new_value=0.6,
            requires_reslice=True,
        )
    )

    assert widget.iso_threshold == pytest.approx(0.6)
    assert emitted == []  # applied under _applying, no echo


def test_inbound_echo_filtered_by_source_id():
    widget, visual_id = _make_widget(initial_render_mode="mip")

    widget._on_appearance_changed(
        AppearanceChangedEvent(
            source_id=widget._id,  # our own echo -> ignored
            visual_id=visual_id,
            field_name="render_mode",
            new_value="iso",
            requires_reslice=True,
        )
    )

    assert widget.render_mode == "mip"  # unchanged


def test_inbound_unrelated_field_ignored():
    widget, visual_id = _make_widget(initial_render_mode="mip")

    widget._on_appearance_changed(
        AppearanceChangedEvent(
            source_id=uuid4(),
            visual_id=visual_id,
            field_name="clim",
            new_value=(0.0, 500.0),
            requires_reslice=False,
        )
    )

    assert widget.render_mode == "mip"  # unchanged
