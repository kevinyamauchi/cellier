"""Tests for the anywidget render-settings panels.

The notebook twins of ``tests/gui/qt/test_render_panels.py``.  Both front
ends draw from the same control spec, so the parity checks live in
``tests/gui/test_render_controls_parity.py`` and this module covers only
what the anywidget layer does on its own: flattened synced traits, the
dotted-path spelling, and the serialised spec the ESM renders from.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("anywidget")

from cellier.controller import CellierController
from cellier.events import (
    RenderConfigChangedEvent,
    RenderConfigUpdateEvent,
)
from cellier.gui._protocol import WidgetView
from cellier.gui.anywidget.render import (
    AnywidgetAmbientOcclusionControls,
    AnywidgetOutlineControls,
    AnywidgetTemporalControls,
)
from cellier.gui.anywidget.render._base import (
    field_name,
    trait_name,
)
from cellier.render._config import (
    AmbientOcclusionConfig,
    OutlineConfig,
    TemporalAccumulationConfig,
)

_PANELS = ("ambient_occlusion", "outline", "temporal")


def _make_panel(kind: str):
    if kind == "ambient_occlusion":
        return AnywidgetAmbientOcclusionControls(AmbientOcclusionConfig())
    if kind == "outline":
        return AnywidgetOutlineControls(OutlineConfig())
    return AnywidgetTemporalControls(TemporalAccumulationConfig())


def _config_for(section: str):
    return {
        "ambient_occlusion": AmbientOcclusionConfig,
        "outline": OutlineConfig,
        "temporal": TemporalAccumulationConfig,
    }[section]()


@pytest.fixture
def controller():
    ctrl = CellierController()
    ctrl.camera_reslice_enabled = False
    yield ctrl
    ctrl.close()


# ---------------------------------------------------------------------------
# Trait naming
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "trait"),
    [
        ("power", "power"),
        ("n_samples", "n_samples"),
        ("selection.inward_thickness", "selection__inward_thickness"),
        ("boundaries.color", "boundaries__color"),
        ("inner_color", "inner_color"),
    ],
)
def test_a_dotted_field_round_trips_through_a_trait_name(field, trait):
    """A dotted config path is not a legal trait name; a double underscore is.

    ``inner_color`` is the case that would break a naive scheme: it carries
    a single underscore of its own and must survive untouched.
    """
    assert trait_name(field) == trait
    assert field_name(trait) == field


# ---------------------------------------------------------------------------
# The shared contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", _PANELS)
def test_a_panel_satisfies_the_widget_contract(kind):
    panel = _make_panel(kind)
    assert isinstance(panel, WidgetView)
    assert panel.widget is panel
    assert panel.subscription_specs()


@pytest.mark.parametrize("kind", _PANELS)
def test_a_panel_subscribes_without_an_entity_id(kind):
    """There is no UUID a render-config change could be keyed by."""
    panel = _make_panel(kind)
    (spec,) = panel.subscription_specs()
    assert spec.event_type is RenderConfigChangedEvent
    assert spec.entity_id is None


@pytest.mark.parametrize("kind", _PANELS)
def test_a_trait_edit_emits_an_update_for_that_section(kind):
    panel = _make_panel(kind)
    emitted: list[RenderConfigUpdateEvent] = []
    panel.changed.connect(emitted.append)

    panel.enabled = not panel.enabled

    assert len(emitted) == 1
    assert emitted[0].section == panel.section
    assert emitted[0].field == "enabled"
    assert emitted[0].value == panel.enabled
    assert emitted[0].source_id == panel._id


def test_a_nested_field_emits_its_dotted_path():
    """The controller's seam takes the dotted path, not the trait name."""
    panel = AnywidgetOutlineControls(OutlineConfig())
    emitted: list[RenderConfigUpdateEvent] = []
    panel.changed.connect(emitted.append)

    panel.selection__inward_thickness = 5

    assert emitted[0].field == "selection.inward_thickness"
    assert emitted[0].value == 5


@pytest.mark.parametrize("kind", _PANELS)
def test_a_foreign_change_lands_in_the_trait(kind):
    panel = _make_panel(kind)
    before = panel.enabled

    panel._on_render_config_changed(
        RenderConfigChangedEvent(
            source_id=uuid4(),
            section=panel.section,
            config=_config_for(panel.section),
            field_name="enabled",
            new_value=not before,
        )
    )

    assert panel.enabled is (not before)


@pytest.mark.parametrize("kind", _PANELS)
def test_a_foreign_change_does_not_re_emit(kind):
    """Applying an inbound value must not send it straight back out."""
    panel = _make_panel(kind)
    emitted: list[RenderConfigUpdateEvent] = []
    panel.changed.connect(emitted.append)

    panel._on_render_config_changed(
        RenderConfigChangedEvent(
            source_id=uuid4(),
            section=panel.section,
            config=_config_for(panel.section),
            field_name="enabled",
            new_value=not panel.enabled,
        )
    )

    assert emitted == []


@pytest.mark.parametrize("kind", _PANELS)
def test_a_panel_ignores_the_echo_of_its_own_change(kind):
    panel = _make_panel(kind)
    before = panel.enabled

    panel._on_render_config_changed(
        RenderConfigChangedEvent(
            source_id=panel._id,
            section=panel.section,
            config=_config_for(panel.section),
            field_name="enabled",
            new_value=not before,
        )
    )

    assert panel.enabled is before


@pytest.mark.parametrize("kind", _PANELS)
def test_a_panel_ignores_another_sections_event(kind):
    panel = _make_panel(kind)
    before = panel.enabled
    other = next(s for s in _PANELS if s != panel.section)

    panel._on_render_config_changed(
        RenderConfigChangedEvent(
            source_id=uuid4(),
            section=other,
            config=_config_for(other),
            field_name="enabled",
            new_value=not before,
        )
    )

    assert panel.enabled is before


@pytest.mark.parametrize("kind", _PANELS)
def test_a_whole_section_replacement_refreshes_every_trait(kind):
    panel = _make_panel(kind)
    replacement = _config_for(panel.section)
    replacement.enabled = not panel.enabled

    panel._on_render_config_changed(
        RenderConfigChangedEvent(
            source_id=uuid4(),
            section=panel.section,
            config=replacement,
            field_name=None,
            new_value=None,
        )
    )

    assert panel.enabled is replacement.enabled


# ---------------------------------------------------------------------------
# The serialised spec the ESM renders from
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", _PANELS)
def test_every_described_control_has_a_trait(kind):
    """The JS looks each control's trait up by name; all of them must exist."""
    panel = _make_panel(kind)
    for control in panel.controls:
        assert panel.has_trait(control["trait"])
        assert control["kind"] in {"bool", "int", "float", "color", "palette"}


@pytest.mark.parametrize("kind", _PANELS)
def test_every_numeric_control_has_a_usable_range(kind):
    """A slider with a null bound renders as an unusable control."""
    panel = _make_panel(kind)
    for control in panel.controls:
        if control["kind"] in {"int", "float"}:
            assert control["min"] is not None, control["field"]
            assert control["max"] is not None, control["field"]
            assert control["step"] > 0


def test_the_ssao_radius_slider_is_sized_from_the_scene():
    """A radius is in scene units, so no fixed maximum would mean anything."""
    panel = AnywidgetAmbientOcclusionControls(
        AmbientOcclusionConfig(), effective_radius=lambda: 12.5
    )
    radius = next(c for c in panel.controls if c["field"] == "radius")
    assert radius["max"] == pytest.approx(50.0)


def test_the_ssao_radius_slider_survives_having_no_canvas():
    """Built before ``add_canvas``, the panel must still describe a range."""
    panel = AnywidgetAmbientOcclusionControls(
        AmbientOcclusionConfig(), effective_radius=lambda: None
    )
    radius = next(c for c in panel.controls if c["field"] == "radius")
    assert radius["max"] > 0


def test_an_auto_radius_arrives_as_none():
    """``radius = None`` *is* the auto mode; the trait has to carry it."""
    panel = AnywidgetAmbientOcclusionControls(AmbientOcclusionConfig(radius=10.0))
    assert panel.radius == pytest.approx(10.0)

    panel._on_render_config_changed(
        RenderConfigChangedEvent(
            source_id=uuid4(),
            section="ambient_occlusion",
            config=AmbientOcclusionConfig(),
            field_name="radius",
            new_value=None,
        )
    )

    assert panel.radius is None


# ---------------------------------------------------------------------------
# Readouts and actions
# ---------------------------------------------------------------------------


def test_the_ssao_panel_reports_the_effective_radius():
    panel = AnywidgetAmbientOcclusionControls(
        AmbientOcclusionConfig(), effective_radius=lambda: 12.5
    )
    assert panel.readouts == [["Radius in use", "12.5 scene units"]]


def test_the_temporal_panel_reports_convergence():
    counts = [0]
    panel = AnywidgetTemporalControls(
        TemporalAccumulationConfig(), frame_count=lambda: counts[0]
    )
    assert panel.readouts == [["State", "restarting"]]

    counts[0] = 3
    panel.refresh_readouts()
    assert "settling" in panel.readouts[0][1]

    counts[0] = 40
    panel.refresh_readouts()
    assert "settled" in panel.readouts[0][1]


def test_the_temporal_reset_action_calls_back():
    """Resetting is an action, not a setting, so it does not use the bus."""
    calls = []
    panel = AnywidgetTemporalControls(
        TemporalAccumulationConfig(),
        frame_count=lambda: 5,
        on_reset=lambda: calls.append(1),
    )
    emitted = []
    panel.changed.connect(emitted.append)
    assert panel.action_label == "Reset history"

    panel._action_clicks += 1

    assert calls == [1]
    assert emitted == []


def test_the_temporal_panel_omits_the_optional_parts_by_default():
    panel = AnywidgetTemporalControls(TemporalAccumulationConfig())
    assert panel.readouts == []
    assert panel.action_label == ""


# ---------------------------------------------------------------------------
# End to end, through the controller
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", _PANELS)
def test_a_panel_edit_reaches_the_render_config(kind, controller):
    panel = _make_panel(kind)
    controller.connect_widget(panel, subscription_specs=panel.subscription_specs())

    panel.enabled = not panel.enabled

    section = getattr(controller.render_config, panel.section)
    assert section.enabled is panel.enabled


@pytest.mark.parametrize("kind", _PANELS)
def test_a_programmatic_change_reaches_the_panel(kind, controller):
    panel = _make_panel(kind)
    controller.connect_widget(panel, subscription_specs=panel.subscription_specs())

    section = getattr(controller.render_config, panel.section)
    controller.update_render_config_field(panel.section, "enabled", not section.enabled)

    assert panel.enabled is section.enabled
