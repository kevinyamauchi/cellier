"""Tests for the Qt render-settings panels.

Three panels -- outlines, ambient occlusion, temporal accumulation -- share
one base, so the bus contract is tested once across all three and the
per-panel tests cover only what each one actually does differently.

These panels differ from the appearance widgets in one way that matters
here: they carry no entity id, because render configuration belongs to the
``RenderManager`` rather than to a scene, a visual or a canvas.  So the
subscription is unfiltered and each panel discards the sections that are
not its own -- which is what ``test_a_panel_ignores_another_sections_event``
pins.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from cellier.controller import CellierController
from cellier.events import RenderConfigChangedEvent, RenderConfigUpdateEvent
from cellier.gui._protocol import WidgetView
from cellier.gui.qt.render import (
    QtAmbientOcclusionControls,
    QtOutlineControls,
    QtTemporalControls,
)
from cellier.render._config import (
    AmbientOcclusionConfig,
    OutlineConfig,
    TemporalAccumulationConfig,
)
from tests.convenience._qt_acceptance import assert_panel_renders, control_labels


@pytest.fixture
def controller():
    ctrl = CellierController()
    ctrl.camera_reslice_enabled = False
    yield ctrl
    ctrl.close()


def _make_panel(kind, qtbot):
    """Build one panel of *kind* over a default config, parented for cleanup."""
    if kind == "ambient_occlusion":
        panel = QtAmbientOcclusionControls(AmbientOcclusionConfig())
    elif kind == "outline":
        panel = QtOutlineControls(OutlineConfig())
    else:
        panel = QtTemporalControls(TemporalAccumulationConfig())
    qtbot.addWidget(panel.widget)
    return panel


_PANELS = ("ambient_occlusion", "outline", "temporal")

#: One control per panel that every panel is known to have, with the value to
#: set and the field it should report.
_ENABLED_FIELD = "enabled"


# ---------------------------------------------------------------------------
# The shared contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", _PANELS)
def test_a_panel_satisfies_the_widget_contract(kind, qtbot):
    """``connect_widget`` wires these the same way it wires a Qt field."""
    panel = _make_panel(kind, qtbot)
    assert isinstance(panel, WidgetView)
    assert panel.widget is not None
    assert panel.subscription_specs()


@pytest.mark.parametrize("kind", _PANELS)
def test_a_panel_subscribes_without_an_entity_id(kind, qtbot):
    """There is no UUID a render-config change could be keyed by."""
    panel = _make_panel(kind, qtbot)
    (spec,) = panel.subscription_specs()
    assert spec.event_type is RenderConfigChangedEvent
    assert spec.entity_id is None


@pytest.mark.parametrize(("kind", "section"), list(zip(_PANELS, _PANELS)))
def test_a_user_edit_emits_an_update_for_that_section(kind, section, qtbot):
    """Toggling a control emits the event the controller's seam consumes."""
    panel = _make_panel(kind, qtbot)
    emitted: list[RenderConfigUpdateEvent] = []
    panel.changed.connect(emitted.append)

    checkbox = _control_for(panel, _ENABLED_FIELD)
    checkbox.setChecked(not checkbox.isChecked())

    assert len(emitted) == 1
    event = emitted[0]
    assert event.section == section
    assert event.field == _ENABLED_FIELD
    assert event.value == checkbox.isChecked()
    assert event.source_id == panel._id


@pytest.mark.parametrize("kind", _PANELS)
def test_a_foreign_change_lands_in_the_control(kind, qtbot):
    """A change made elsewhere -- code, another widget -- shows up here."""
    panel = _make_panel(kind, qtbot)
    checkbox = _control_for(panel, _ENABLED_FIELD)
    before = checkbox.isChecked()

    panel._on_render_config_changed(
        RenderConfigChangedEvent(
            source_id=uuid4(),
            section=panel.section,
            config=_config_for(panel.section),
            field_name=_ENABLED_FIELD,
            new_value=not before,
        )
    )

    assert checkbox.isChecked() is (not before)


@pytest.mark.parametrize("kind", _PANELS)
def test_a_foreign_change_does_not_re_emit(kind, qtbot):
    """Applying an inbound value must not send it straight back out.

    Without the blocked signals this is an infinite round trip, and in
    practice it fights the user mid-drag.
    """
    panel = _make_panel(kind, qtbot)
    emitted: list[RenderConfigUpdateEvent] = []
    panel.changed.connect(emitted.append)

    panel._on_render_config_changed(
        RenderConfigChangedEvent(
            source_id=uuid4(),
            section=panel.section,
            config=_config_for(panel.section),
            field_name=_ENABLED_FIELD,
            new_value=True,
        )
    )

    assert emitted == []


@pytest.mark.parametrize("kind", _PANELS)
def test_a_panel_ignores_the_echo_of_its_own_change(kind, qtbot):
    """The panel's own source id means the value is already in the control."""
    panel = _make_panel(kind, qtbot)
    checkbox = _control_for(panel, _ENABLED_FIELD)
    before = checkbox.isChecked()

    panel._on_render_config_changed(
        RenderConfigChangedEvent(
            source_id=panel._id,
            section=panel.section,
            config=_config_for(panel.section),
            field_name=_ENABLED_FIELD,
            new_value=not before,
        )
    )

    assert checkbox.isChecked() is before


@pytest.mark.parametrize("kind", _PANELS)
def test_a_panel_ignores_another_sections_event(kind, qtbot):
    """The subscription is unfiltered, so the panel filters it itself."""
    panel = _make_panel(kind, qtbot)
    checkbox = _control_for(panel, _ENABLED_FIELD)
    before = checkbox.isChecked()
    other = next(s for s in _PANELS if s != panel.section)

    panel._on_render_config_changed(
        RenderConfigChangedEvent(
            source_id=uuid4(),
            section=other,
            config=_config_for(other),
            field_name=_ENABLED_FIELD,
            new_value=not before,
        )
    )

    assert checkbox.isChecked() is before


@pytest.mark.parametrize("kind", _PANELS)
def test_a_whole_section_replacement_refreshes_every_control(kind, qtbot):
    """``field_name=None`` means "everything changed"; re-read it all."""
    panel = _make_panel(kind, qtbot)
    replacement = _config_for(panel.section)
    replacement.enabled = not _config_for(panel.section).enabled

    panel._on_render_config_changed(
        RenderConfigChangedEvent(
            source_id=uuid4(),
            section=panel.section,
            config=replacement,
            field_name=None,
            new_value=None,
        )
    )

    assert _control_for(panel, _ENABLED_FIELD).isChecked() is replacement.enabled


@pytest.mark.parametrize("kind", _PANELS)
def test_a_panel_renders(kind, qtbot):
    """A panel of correctly-titled but empty controls would fail this."""
    panel = _make_panel(kind, qtbot)
    assert_panel_renders(panel.widget)


@pytest.mark.parametrize("kind", _PANELS)
def test_every_control_a_panel_shows_is_settable(kind, qtbot):
    """No control may name a field the controller refuses to set.

    A panel with a control for an unroutable field looks like it works and
    raises when touched.
    """
    from cellier.controller import _RENDER_CONFIG_ROUTES

    panel = _make_panel(kind, qtbot)
    for field in panel._appliers:
        assert (panel.section, field) in _RENDER_CONFIG_ROUTES


# ---------------------------------------------------------------------------
# End to end, through the controller
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", _PANELS)
def test_a_panel_edit_reaches_the_render_config(kind, controller, qtbot):
    """The whole round trip: control -> bus -> seam -> config."""
    panel = _make_panel(kind, qtbot)
    controller.connect_widget(panel, subscription_specs=panel.subscription_specs())

    checkbox = _control_for(panel, _ENABLED_FIELD)
    checkbox.setChecked(not checkbox.isChecked())

    section = getattr(controller.render_config, panel.section)
    assert section.enabled is checkbox.isChecked()


@pytest.mark.parametrize("kind", _PANELS)
def test_a_programmatic_change_reaches_the_panel(kind, controller, qtbot):
    """The other direction, which is what routing the properties buys.

    ``controller.ambient_occlusion_power = 0.5`` in a notebook cell has to move the
    slider, or the panel is lying about the state of the renderer.
    """
    panel = _make_panel(kind, qtbot)
    controller.connect_widget(panel, subscription_specs=panel.subscription_specs())

    section = getattr(controller.render_config, panel.section)
    controller.update_render_config_field(
        panel.section, _ENABLED_FIELD, not section.enabled
    )

    assert _control_for(panel, _ENABLED_FIELD).isChecked() is section.enabled


# ---------------------------------------------------------------------------
# Per-panel specifics
# ---------------------------------------------------------------------------


def test_the_ssao_panel_shows_the_effective_radius(qtbot):
    """The number that makes a radius in scene units mean anything."""
    panel = QtAmbientOcclusionControls(
        AmbientOcclusionConfig(), effective_radius=lambda: 12.5
    )
    qtbot.addWidget(panel.widget)
    assert "12.5" in " ".join(_all_label_text(panel.widget))


def test_the_ssao_radius_readout_survives_having_no_canvas(qtbot):
    """A panel built before ``add_canvas`` must still render."""
    panel = QtAmbientOcclusionControls(
        AmbientOcclusionConfig(), effective_radius=lambda: None
    )
    qtbot.addWidget(panel.widget)
    assert "no canvas yet" in " ".join(_all_label_text(panel.widget))


def test_the_ssao_auto_radius_checkbox_writes_radius_none(qtbot):
    """``radius = None`` *is* the auto mode; there is no separate flag."""
    panel = QtAmbientOcclusionControls(AmbientOcclusionConfig(radius=10.0))
    qtbot.addWidget(panel.widget)
    emitted: list[RenderConfigUpdateEvent] = []
    panel.changed.connect(emitted.append)

    auto = _auto_radius_checkbox(panel)
    auto.setChecked(True)

    assert emitted[-1].field == "radius"
    assert emitted[-1].value is None


def test_an_inbound_auto_radius_moves_the_checkbox(qtbot):
    """The checkbox follows the field rather than owning it."""
    panel = QtAmbientOcclusionControls(AmbientOcclusionConfig(radius=10.0))
    qtbot.addWidget(panel.widget)
    auto = _auto_radius_checkbox(panel)
    assert auto.isChecked() is False

    panel._on_render_config_changed(
        RenderConfigChangedEvent(
            source_id=uuid4(),
            section="ambient_occlusion",
            config=AmbientOcclusionConfig(),
            field_name="radius",
            new_value=None,
        )
    )

    assert auto.isChecked() is True


def test_the_outline_panel_has_no_selection_colour_control(qtbot):
    """It would do nothing: that layer's colour comes from the palette slot."""
    panel = QtOutlineControls(OutlineConfig())
    qtbot.addWidget(panel.widget)
    assert "selection.color" not in panel._appliers
    assert "boundaries.color" in panel._appliers


def test_the_outline_panel_covers_both_layers(qtbot):
    """Both layers are independently reachable, thickness and enable alike."""
    panel = QtOutlineControls(OutlineConfig())
    qtbot.addWidget(panel.widget)
    for layer in ("boundaries", "selection"):
        for field in ("enabled", "inward_thickness", "outward_thickness"):
            assert f"{layer}.{field}" in panel._appliers


def test_the_temporal_panel_reports_convergence(qtbot):
    """The one thing about this pass a user cannot otherwise see."""
    counts = [0]
    panel = QtTemporalControls(
        TemporalAccumulationConfig(), frame_count=lambda: counts[0]
    )
    qtbot.addWidget(panel.widget)
    assert "restarting" in " ".join(_all_label_text(panel.widget))

    counts[0] = 3
    panel.refresh_readouts()
    assert "settling" in " ".join(_all_label_text(panel.widget))

    counts[0] = 40
    panel.refresh_readouts()
    assert "settled" in " ".join(_all_label_text(panel.widget))


def test_the_temporal_reset_button_calls_back(qtbot):
    """Resetting is an action, not a setting, so it does not use the bus."""
    calls = []
    panel = QtTemporalControls(
        TemporalAccumulationConfig(),
        frame_count=lambda: 5,
        on_reset=lambda: calls.append(1),
    )
    qtbot.addWidget(panel.widget)
    emitted = []
    panel.changed.connect(emitted.append)

    _reset_button(panel).click()

    assert calls == [1]
    assert emitted == []


def test_the_temporal_panel_omits_the_optional_parts_by_default(qtbot):
    """A panel built with neither callback still works."""
    panel = QtTemporalControls(TemporalAccumulationConfig())
    qtbot.addWidget(panel.widget)
    assert panel._readouts == []
    assert set(panel._appliers) == {"enabled", "blend_weight"}


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (
            "ambient_occlusion",
            ["Strength", "Power", "Scene fraction", "Radius"],
        ),
        ("outline", ["Inward", "Outward", "Color", "Thickness"]),
        ("temporal", ["Blend weight"]),
    ],
)
def test_the_panels_name_their_controls(kind, expected, qtbot):
    """A rename or a reordering shows up as a diff rather than by eye."""
    panel = _make_panel(kind, qtbot)
    labels = control_labels(panel.widget)
    for name in expected:
        assert name in labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_for(section: str):
    return {
        "ambient_occlusion": AmbientOcclusionConfig,
        "outline": OutlineConfig,
        "temporal": TemporalAccumulationConfig,
    }[section]()


def _control_for(panel, field: str):
    """Recover the Qt control behind a registered field applier."""
    applier = panel._appliers[field]
    return applier.__closure__[0].cell_contents


def _all_label_text(widget) -> list[str]:
    from PySide6.QtWidgets import QLabel

    return [child.text() for child in widget.findChildren(QLabel)]


def _auto_radius_checkbox(panel):
    from PySide6.QtWidgets import QCheckBox

    return next(
        box
        for box in panel.widget.findChildren(QCheckBox)
        if box.text() == "Derive from scene size"
    )


def _reset_button(panel):
    from PySide6.QtWidgets import QPushButton

    return next(
        button
        for button in panel.widget.findChildren(QPushButton)
        if button.text() == "Reset history"
    )


# ---------------------------------------------------------------------------
# Naming and control shape
#
# Every assertion here replaced something a user had to work out by looking.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", _PANELS)
def test_the_master_switch_is_named_for_its_field(kind, qtbot):
    """A checkbox restating the group it sits in wastes the only line there.

    "Outlines" inside a group titled "Outlines" told the reader nothing.
    "Enabled" says what it does *and* matches the field it writes, which is
    the property the whole naming pass is after.
    """
    panel = _make_panel(kind, qtbot)
    assert _control_for(panel, "enabled").text() == "Enabled"


def test_the_layer_switches_are_named_for_their_field(qtbot):
    panel = QtOutlineControls(OutlineConfig())
    qtbot.addWidget(panel.widget)
    for layer in ("boundaries", "selection"):
        assert _control_for(panel, f"{layer}.enabled").text() == "Enabled"


@pytest.mark.parametrize("kind", _PANELS)
def test_every_tooltip_names_the_attribute_it_writes(kind, qtbot):
    """A readable label and a findable one, instead of a trade between them.

    Generated from the spec rather than written by hand, so it cannot drift
    from the field -- and so a group that is not real nesting gives itself
    away: "Contrast band / Thickness" reads
    ``render_config.outline.inner_thickness``.
    """
    from cellier.gui._render_controls import RENDER_CONTROLS

    panel = _make_panel(kind, qtbot)
    for control in RENDER_CONTROLS[kind]:
        tip = panel._tooltip(control)
        assert tip.endswith(f"render_config.{kind}.{control.field}"), control.field

    # And it reaches a real widget, not only the generator.
    assert f"render_config.{kind}.enabled" in (_control_for(panel, "enabled").toolTip())


def test_the_palette_is_named_by_its_group_not_by_a_row(qtbot):
    """The group is "Outline colors"; a row repeating it says nothing."""
    from cellier.gui._render_controls import RENDER_CONTROLS

    panel = QtOutlineControls(OutlineConfig())
    qtbot.addWidget(panel.widget)
    spec = next(c for c in RENDER_CONTROLS["outline"] if c.field == "palette")
    assert spec.label == ""
    assert spec.group == "Outline colors"
    assert "Outline colors" in _group_titles(panel)
    assert "Selection colors" not in _all_label_text(panel.widget)


def test_a_colour_swatch_opens_the_picker_itself(qtbot):
    """No separate button: a swatch is the most clickable thing on the row."""
    from PySide6.QtWidgets import QPushButton

    panel = QtOutlineControls(OutlineConfig())
    qtbot.addWidget(panel.widget)
    buttons = [b.text() for b in panel.widget.findChildren(QPushButton)]
    assert "Choose..." not in buttons


def test_alpha_gets_a_row_of_its_own(qtbot):
    """Sharing the swatch's row left it a few pixels wide and clipped.

    One row per colour control, and this panel has two of them.
    """
    panel = QtOutlineControls(OutlineConfig())
    qtbot.addWidget(panel.widget)
    assert _all_label_text(panel.widget).count("Alpha") == 2


def _group_titles(panel) -> list[str]:
    from PySide6.QtWidgets import QGroupBox

    return [box.title() for box in panel.widget.findChildren(QGroupBox)]
