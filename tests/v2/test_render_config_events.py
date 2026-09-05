"""Tests for the render-config event pair and the controller seam.

Render configuration -- outlines, ambient occlusion, temporal accumulation --
belongs to the ``RenderManager`` rather than to a scene, a visual or a
canvas, so it travels on its own event pair with no entity id, and every
subscriber filters on ``section`` itself.

The seam under test is ``CellierController.update_render_config_field``: one
method that writes the model, pushes the change to the GPU by whichever
route the field needs, and emits the outgoing event.  The route table it
consults is the single place recording which fields recompile a shader,
which is exactly the thing a widget must not have to know.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from cellier.controller import (
    _RENDER_CONFIG_ROUTES,
    RENDER_CONFIG_SECTIONS,
    CellierController,
)
from cellier.events import RenderConfigChangedEvent, RenderConfigUpdateEvent


@pytest.fixture
def controller():
    """A controller with camera-driven reslicing off."""
    ctrl = CellierController()
    ctrl.camera_reslice_enabled = False
    yield ctrl
    ctrl.close()


@pytest.fixture
def seen(controller):
    """Every ``RenderConfigChangedEvent`` the controller emits."""
    events: list[RenderConfigChangedEvent] = []
    controller._outgoing_events.subscribe(
        RenderConfigChangedEvent, events.append, owner_id=uuid4()
    )
    return events


# ---------------------------------------------------------------------------
# The route table
# ---------------------------------------------------------------------------

#: One in-range value per settable field, so the whole table can be exercised.
_VALUES: dict[tuple[str, str], object] = {
    ("outline", "enabled"): True,
    ("outline", "boundaries.enabled"): False,
    ("outline", "selection.enabled"): False,
    ("outline", "boundaries.inward_thickness"): 3,
    ("outline", "boundaries.outward_thickness"): 3,
    ("outline", "selection.inward_thickness"): 4,
    ("outline", "selection.outward_thickness"): 4,
    ("outline", "inner_thickness"): 5,
    ("outline", "boundaries.color"): (1.0, 0.0, 0.0, 1.0),
    ("outline", "inner_color"): (0.0, 1.0, 0.0, 1.0),
    ("outline", "palette"): [(0.0, 0.0, 1.0, 1.0)],
    ("ambient_occlusion", "enabled"): True,
    ("ambient_occlusion", "n_samples"): 32,
    ("ambient_occlusion", "blur_radius"): 4,
    ("ambient_occlusion", "radius"): 12.0,
    ("ambient_occlusion", "auto_radius_fraction"): 0.05,
    ("ambient_occlusion", "bias"): 0.1,
    ("ambient_occlusion", "strength"): 0.5,
    ("ambient_occlusion", "power"): 2.0,
    ("temporal", "enabled"): False,
    ("temporal", "blend_weight"): 0.25,
}


def test_every_settable_field_has_a_test_value():
    """The parametrised test below covers the table, not a subset of it."""
    assert set(_VALUES) == set(_RENDER_CONFIG_ROUTES)


@pytest.mark.parametrize(("section", "field"), sorted(_RENDER_CONFIG_ROUTES))
def test_each_field_reaches_the_model_and_emits(controller, seen, section, field):
    """Every field in the route table writes through and announces itself."""
    value = _VALUES[(section, field)]

    controller.update_render_config_field(section, field, value)

    target = getattr(controller.render_config, section)
    *parents, leaf = field.split(".")
    for name in parents:
        target = getattr(target, name)
    assert getattr(target, leaf) == value

    assert len(seen) == 1
    event = seen[0]
    assert event.section == section
    assert event.field_name == field
    assert event.new_value == value
    # The whole section, not just the delta: a subscriber holding a snapshot
    # cannot reconstruct the rest of an OutlineConfig from one field.
    assert event.config is getattr(controller.render_config, section)


def test_the_selection_layer_colour_is_not_settable(controller):
    """It exists on the shared layer model and does nothing on that layer.

    The selection layer takes its colour from the palette slot carried in
    the LUT.  Recording that here once is what lets every GUI simply not
    draw a control for it.
    """
    assert ("outline", "selection.color") not in _RENDER_CONFIG_ROUTES
    with pytest.raises(ValueError, match="not a settable field"):
        controller.update_render_config_field(
            "outline", "selection.color", (1.0, 0.0, 0.0, 1.0)
        )


# ---------------------------------------------------------------------------
# The properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prop", "value"),
    [
        ("outline_enabled", True),
        ("outline_boundaries_enabled", False),
        ("outline_selection_enabled", False),
        ("ambient_occlusion_enabled", True),
        ("ambient_occlusion_radius", 8.0),
        ("ambient_occlusion_auto_radius_fraction", 0.04),
        ("ambient_occlusion_strength", 0.25),
        ("ambient_occlusion_power", 2.0),
        ("ambient_occlusion_bias", 0.2),
        ("ambient_occlusion_n_samples", 24),
        ("ambient_occlusion_blur_radius", 1),
        ("temporal_enabled", False),
        ("temporal_blend_weight", 0.5),
    ],
)
def test_the_convenience_properties_round_trip_and_emit(controller, seen, prop, value):
    """Each property reads back what it was given, and announces the change.

    Routing the properties through the same seam is what makes a
    programmatic ``controller.ambient_occlusion_power = 0.5`` reach a subscribed widget;
    before this they wrote straight to the render manager and no GUI
    noticed.
    """
    setattr(controller, prop, value)
    assert getattr(controller, prop) == value
    assert len(seen) == 1


def test_the_whole_ssao_config_is_reachable_without_a_private_attribute():
    """No setting should need ``controller._render_manager`` to change.

    The demos used to reach through it for ``n_samples``, ``blur_radius``
    and the temporal pair, and into ``canvas._ssao_pass`` for
    ``auto_radius_fraction``.
    """
    from cellier.render._config import AmbientOcclusionConfig

    settable = {
        field
        for section, field in _RENDER_CONFIG_ROUTES
        if section == "ambient_occlusion"
    }
    assert set(AmbientOcclusionConfig.model_fields) == settable


def test_the_whole_temporal_config_is_reachable():
    from cellier.render._config import TemporalAccumulationConfig

    settable = {
        field for section, field in _RENDER_CONFIG_ROUTES if section == "temporal"
    }
    assert set(TemporalAccumulationConfig.model_fields) == settable


# ---------------------------------------------------------------------------
# The incoming bus
# ---------------------------------------------------------------------------


def test_a_widget_update_event_drives_the_config(controller):
    """The route a GUI actually takes: emit, and the config follows."""
    controller.incoming_events.emit(
        RenderConfigUpdateEvent(
            source_id=uuid4(), section="ambient_occlusion", field="power", value=3.0
        )
    )
    assert controller.ambient_occlusion_power == 3.0


def test_the_widget_source_id_is_stamped_on_the_echo(controller, seen):
    """So a widget can ignore the echo of its own change.

    Without this every widget would fight the user: the change comes back
    round the bus and resets the control mid-drag.
    """
    widget_id = uuid4()
    controller.incoming_events.emit(
        RenderConfigUpdateEvent(
            source_id=widget_id, section="temporal", field="blend_weight", value=0.3
        )
    )
    assert seen[-1].source_id == widget_id


def test_a_controller_driven_change_is_stamped_with_the_controller(controller, seen):
    controller.ambient_occlusion_power = 2.0
    assert seen[-1].source_id == controller._id


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_an_unknown_section_raises_with_a_suggestion(controller):
    with pytest.raises(ValueError, match="Did you mean 'ambient_occlusion'"):
        controller.update_render_config_field("ambient_occlusio", "power", 1.0)


def test_an_unknown_field_raises_with_a_suggestion(controller):
    with pytest.raises(ValueError, match="Did you mean 'power'"):
        controller.update_render_config_field("ambient_occlusion", "powr", 1.0)


def test_the_sections_tuple_matches_the_route_table(controller):
    assert set(RENDER_CONFIG_SECTIONS) == {s for s, _f in _RENDER_CONFIG_ROUTES}


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("ambient_occlusion", "power", -1.0),
        ("ambient_occlusion", "n_samples", 200),
        ("ambient_occlusion", "strength", 2.0),
        ("temporal", "blend_weight", 0.0),
        ("outline", "selection.inward_thickness", -1),
    ],
)
def test_an_out_of_range_value_is_rejected_before_it_reaches_the_gpu(
    controller, seen, section, field, value
):
    """The config models validate on assignment, so a slider cannot lie.

    Without this the declared bounds on the config fields would be
    documentation only, and a bad value would reach a shader uniform.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        controller.update_render_config_field(section, field, value)
    assert seen == [], "a rejected value must not announce itself"
