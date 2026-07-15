"""Tests for the anywidget ``AnywidgetAABBWidget`` AABB-controls widget."""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("anywidget")

from cellier.events import AABBChangedEvent  # noqa: E402
from cellier.gui.anywidget.visuals._aabb import AnywidgetAABBWidget  # noqa: E402


def _make_widget(**kwargs):
    visual_id = uuid4()
    return AnywidgetAABBWidget(visual_id, **kwargs), visual_id


def test_instantiate_smoke():
    widget, _visual_id = _make_widget(
        initial_enabled=True, initial_line_width=3.0, initial_color="#ff00ff"
    )
    assert widget.enabled is True
    assert widget.line_width == pytest.approx(3.0)
    assert widget.color == "#ff00ff"


def test_trait_edit_emits_update_event():
    widget, visual_id = _make_widget()
    emitted = []
    widget.changed.connect(emitted.append)

    widget.line_width = 6.5

    assert len(emitted) == 1
    event = emitted[0]
    assert event.source_id == widget._id
    assert event.visual_id == visual_id
    assert event.field == "line_width"
    assert event.value == pytest.approx(6.5)


def test_enabled_edit_emits_update_event():
    widget, _visual_id = _make_widget(initial_enabled=False)
    emitted = []
    widget.changed.connect(emitted.append)

    widget.enabled = True

    assert len(emitted) == 1
    assert emitted[0].field == "enabled"
    assert emitted[0].value is True


def test_inbound_change_updates_trait_without_reemit():
    widget, visual_id = _make_widget()
    emitted = []
    widget.changed.connect(emitted.append)

    widget._on_aabb_changed(
        AABBChangedEvent(
            source_id=uuid4(),  # not the widget -> applied
            visual_id=visual_id,
            field_name="color",
            new_value="#00ff00",
        )
    )

    assert widget.color == "#00ff00"
    assert emitted == []  # applied under _applying, no echo


def test_inbound_echo_filtered_by_source_id():
    widget, visual_id = _make_widget(initial_color="#ffffff")

    widget._on_aabb_changed(
        AABBChangedEvent(
            source_id=widget._id,  # our own echo -> ignored
            visual_id=visual_id,
            field_name="color",
            new_value="#00ff00",
        )
    )

    assert widget.color == "#ffffff"  # unchanged
