"""Tests for the toolkit-neutral appearance-field specs."""

from __future__ import annotations

from uuid import uuid4

import pytest

from cellier.events import (
    AppearanceChangedEvent,
    VisualVisibilityChangedEvent,
)
from cellier.gui._appearance_fields import (
    NO_MATCH,
    AppearanceFieldSpec,
    VisibleFieldSpec,
    appearance_field_spec,
    normalize_visual_ids,
)

# ---------------------------------------------------------------------------
# Spec selection
# ---------------------------------------------------------------------------


def test_ordinary_field_gets_the_appearance_changed_spec():
    spec = appearance_field_spec("wireframe", "Wireframe")
    assert type(spec) is AppearanceFieldSpec
    assert spec.name == "wireframe"
    assert spec.label == "Wireframe"
    assert spec.inbound_event_type is AppearanceChangedEvent


def test_visible_gets_its_own_spec():
    """``visible`` is routed to a different bus event by the controller.

    ``Controller._make_appearance_handler`` turns a ``visible`` change into a
    ``VisualVisibilityChangedEvent``, not an ``AppearanceChangedEvent``, so a
    widget subscribing to the latter would never see it.
    """
    spec = appearance_field_spec("visible", "Visible")
    assert type(spec) is VisibleFieldSpec
    assert spec.inbound_event_type is VisualVisibilityChangedEvent
    # The label still comes from the caller, not baked into the special spec.
    assert appearance_field_spec("visible", "Show").label == "Show"


# ---------------------------------------------------------------------------
# Value extraction
# ---------------------------------------------------------------------------


def test_ordinary_spec_matches_on_field_name():
    spec = appearance_field_spec("wireframe", "Wireframe")
    event = AppearanceChangedEvent(
        source_id=uuid4(),
        visual_id=uuid4(),
        field_name="wireframe",
        new_value=True,
        requires_reslice=False,
    )
    assert spec.inbound_value(event) is True


def test_ordinary_spec_ignores_another_field():
    spec = appearance_field_spec("wireframe", "Wireframe")
    event = AppearanceChangedEvent(
        source_id=uuid4(),
        visual_id=uuid4(),
        field_name="shininess",
        new_value=12.0,
        requires_reslice=False,
    )
    assert spec.inbound_value(event) is NO_MATCH


def test_visible_spec_reads_the_visible_attribute():
    spec = appearance_field_spec("visible", "Visible")
    event = VisualVisibilityChangedEvent(
        source_id=uuid4(), visual_id=uuid4(), visible=False
    )
    assert spec.inbound_value(event) is False


# ---------------------------------------------------------------------------
# visual_id normalisation
# ---------------------------------------------------------------------------


def test_normalize_single_uuid():
    visual_id = uuid4()
    assert normalize_visual_ids(visual_id) == (visual_id,)


@pytest.mark.parametrize("container", [list, tuple])
def test_normalize_sequence(container):
    ids = [uuid4(), uuid4(), uuid4()]
    assert normalize_visual_ids(container(ids)) == tuple(ids)


def test_normalize_rejects_an_empty_sequence():
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_visual_ids([])
