"""The two render-settings front ends draw the same controls.

Labels, ranges and tooltips live in
``cellier.gui._render_controls.RENDER_CONTROLS`` precisely so a change
lands in both toolkits at once.  That only holds if both actually build
from it, which is what this module checks: a control added to one front end
by hand, or a spec entry one of them silently drops, fails here rather than
in a notebook six months later.
"""

from __future__ import annotations

import pytest

from cellier.controller import _RENDER_CONFIG_ROUTES
from cellier.gui._render_controls import (
    RENDER_CONTROLS,
    RENDER_SECTION_TITLES,
)

_SECTIONS = ("ambient_occlusion", "outline", "temporal")


def _qt_panel(section: str):
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

    if section == "ambient_occlusion":
        return QtAmbientOcclusionControls(AmbientOcclusionConfig())
    if section == "outline":
        return QtOutlineControls(OutlineConfig())
    return QtTemporalControls(TemporalAccumulationConfig())


def _anywidget_panel(section: str):
    from cellier.gui.anywidget.render import (
        AnywidgetAmbientOcclusionControls,
        AnywidgetOutlineControls,
        AnywidgetTemporalControls,
    )
    from cellier.render._config import (
        AmbientOcclusionConfig,
        OutlineConfig,
        TemporalAccumulationConfig,
    )

    if section == "ambient_occlusion":
        return AnywidgetAmbientOcclusionControls(AmbientOcclusionConfig())
    if section == "outline":
        return AnywidgetOutlineControls(OutlineConfig())
    return AnywidgetTemporalControls(TemporalAccumulationConfig())


# ---------------------------------------------------------------------------
# The spec itself
# ---------------------------------------------------------------------------


def test_every_section_has_controls_and_a_title():
    assert set(RENDER_CONTROLS) == set(_SECTIONS)
    assert set(RENDER_SECTION_TITLES) == set(_SECTIONS)


@pytest.mark.parametrize("section", _SECTIONS)
def test_every_spec_field_is_settable(section):
    """A control over an unroutable field looks like it works and then raises."""
    for control in RENDER_CONTROLS[section]:
        assert (section, control.field) in _RENDER_CONFIG_ROUTES, control.field


@pytest.mark.parametrize("section", _SECTIONS)
def test_every_settable_field_has_a_control(section):
    """The other direction: nothing the controller can set is unreachable.

    This is the check that would have caught ``auto_radius_fraction``,
    which for a long time had no route above the occlusion pass at all and
    so could only be changed by reaching into a private canvas attribute.
    """
    routed = {field for sect, field in _RENDER_CONFIG_ROUTES if sect == section}
    shown = {control.field for control in RENDER_CONTROLS[section]}
    # ``selection.color`` is the deliberate exception in the other
    # direction: unroutable *and* undrawn.
    assert routed - shown == set()


@pytest.mark.parametrize("section", _SECTIONS)
def test_no_control_is_listed_twice(section):
    fields = [control.field for control in RENDER_CONTROLS[section]]
    assert len(fields) == len(set(fields))


@pytest.mark.parametrize("section", _SECTIONS)
def test_grouped_controls_are_contiguous(section):
    """Both front ends draw one box per run of a group, so runs must not split.

    A group appearing twice would render as two boxes with the same
    heading, which reads as a bug rather than as a design.
    """
    seen: list[str] = []
    previous = object()
    for control in RENDER_CONTROLS[section]:
        group = control.group or ""
        if group != previous:
            assert group not in seen, f"{group!r} is split into two runs"
            seen.append(group)
            previous = group


@pytest.mark.parametrize("section", _SECTIONS)
def test_numeric_controls_declare_a_minimum(section):
    """A slider needs a floor.  Only the radius may omit its ceiling."""
    for control in RENDER_CONTROLS[section]:
        if control.kind not in {"int", "float"}:
            continue
        assert control.minimum is not None, control.field
        if control.maximum is None:
            assert control.field == "radius", control.field


# ---------------------------------------------------------------------------
# Both front ends against the spec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("section", _SECTIONS)
def test_the_qt_panel_draws_every_spec_control(section, qtbot):
    panel = _qt_panel(section)
    qtbot.addWidget(panel.widget)
    assert set(panel._appliers) == {c.field for c in RENDER_CONTROLS[section]}


@pytest.mark.parametrize("section", _SECTIONS)
def test_the_anywidget_panel_draws_every_spec_control(section):
    pytest.importorskip("anywidget")
    panel = _anywidget_panel(section)
    assert {c["field"] for c in panel.controls} == {
        c.field for c in RENDER_CONTROLS[section]
    }


@pytest.mark.parametrize("section", _SECTIONS)
def test_both_front_ends_show_the_same_controls(section, qtbot):
    """The point of the shared spec, stated as one assertion."""
    pytest.importorskip("anywidget")
    qt_panel = _qt_panel(section)
    qtbot.addWidget(qt_panel.widget)
    any_panel = _anywidget_panel(section)

    assert set(qt_panel._appliers) == {c["field"] for c in any_panel.controls}


@pytest.mark.parametrize("section", _SECTIONS)
def test_both_front_ends_agree_on_the_title(section, qtbot):
    pytest.importorskip("anywidget")
    qt_panel = _qt_panel(section)
    qtbot.addWidget(qt_panel.widget)
    assert qt_panel.title == RENDER_SECTION_TITLES[section]
    assert _anywidget_panel(section).title == RENDER_SECTION_TITLES[section]
