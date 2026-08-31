"""The five control types stage 4 adds, in both toolkits.

Stage 4 of ``plans/convenience_cleanup.md`` (section 10).  The layer-1 bus
contract was already covered against a stub subclass by
``tests/gui/test_appearance_fields.py`` and the two ``*_visible_toggle``
modules; what is new here is **layer 2** -- one class per control type,
implementing the ``_build`` / ``_read`` / ``_apply`` seam -- and the layer-3
classes, which section 10.6 predicted would need only thin tests because the
behaviour is the base's.

Both toolkits live in one module and are asserted **against each other**: the
whole point of generating the 44 field classes from one catalog is that a
field cannot exist on one side and not the other, and that is worth a test
rather than a convention.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from cellier.gui._appearance_fields import (
    as_rgba,
    field_bounds,
    hex_to_rgba,
    literal_choices,
    rgba_to_hex,
)

pytest.importorskip("qtpy")
pytest.importorskip("superqt")

import cellier.gui.anywidget.visuals as anywidget_visuals
import cellier.gui.qt.visuals as qt_visuals

# (field-class stem, field name, label) -- the section 10.4 catalog, minus
# ``visible`` (stage 0) and ``lod_bias`` (an existing widget, not a new class).
CATALOG = [
    ("OpacitySlider", "opacity", "Opacity"),
    ("LabelsRenderModeCombo", "render_mode", "Render mode"),
    ("SaltSpin", "salt", "Salt"),
    ("BackgroundLabelSpin", "background_label", "Background label"),
    ("UniformColorPicker", "color", "Color"),
    ("SideCombo", "side", "Side"),
    ("WireframeToggle", "wireframe", "Wireframe"),
    ("WireframeThicknessSpin", "wireframe_thickness", "Wireframe thickness"),
    ("ShininessSpin", "shininess", "Shininess"),
    ("FlatShadingToggle", "flat_shading", "Flat shading"),
    ("SizeSpin", "size", "Size"),
    ("SizeSpaceCombo", "size_space", "Size space"),
    ("ThicknessSpin", "thickness", "Thickness"),
    ("ThicknessSpaceCombo", "thickness_space", "Thickness space"),
    ("NodeVisibleToggle", "node_visible", "Nodes visible"),
    ("NodeColorPicker", "node_color", "Node color"),
    ("NodeSizeSpin", "node_size", "Node size"),
    ("NodeSizeSpaceCombo", "node_size_space", "Node size space"),
    ("EdgeVisibleToggle", "edge_visible", "Edges visible"),
    ("EdgeColorPicker", "edge_color", "Edge color"),
    ("EdgeThicknessSpin", "edge_thickness", "Edge thickness"),
    ("EdgeThicknessSpaceCombo", "edge_thickness_space", "Edge thickness space"),
]


def _qt(stem):
    return getattr(qt_visuals, f"Qt{stem}")


def _any(stem):
    return getattr(anywidget_visuals, f"Anywidget{stem}")


# ---------------------------------------------------------------------------
# Layer 3: the catalog, and that the two toolkits agree about it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("stem", "field", "label"), CATALOG)
def test_both_toolkits_bind_the_same_field_and_label(stem, field, label):
    """Layer 3 is a field name and a label; that is the whole class."""
    for cls in (_qt(stem), _any(stem)):
        assert cls._field == field
        assert cls._label == label


@pytest.mark.parametrize(("stem", "field", "_label"), CATALOG)
def test_both_toolkits_share_the_same_default_value(stem, field, _label):
    """A default differing per toolkit would show as a different initial UI."""
    assert _qt(stem)._default_value == _any(stem)._default_value


@pytest.mark.parametrize(("stem", "_field", "_label"), CATALOG)
def test_every_field_class_is_exported_from_both_packages(stem, _field, _label):
    assert f"Qt{stem}" in qt_visuals.__all__
    assert f"Anywidget{stem}" in anywidget_visuals.__all__


@pytest.mark.parametrize(("stem", "field", "_label"), CATALOG)
def test_the_field_name_matches_a_real_appearance_model_field(stem, field, _label):
    """A control for a field no model has would render and write nothing.

    Section 10.1's rule runs the other way too: a widget must exist for every
    valid name, and every widget must name a real field.
    """
    from cellier.visuals._base_visual import BaseAppearance
    from cellier.visuals._graph_memory import GraphAppearance
    from cellier.visuals._label_memory import BaseLabelsAppearance
    from cellier.visuals._lines_memory import LinesMemoryAppearance
    from cellier.visuals._mesh_memory import MeshFlatAppearance, MeshPhongAppearance
    from cellier.visuals._points_memory import PointsMarkerAppearance

    models = (
        BaseAppearance,
        BaseLabelsAppearance,
        MeshFlatAppearance,
        MeshPhongAppearance,
        PointsMarkerAppearance,
        LinesMemoryAppearance,
        GraphAppearance,
    )
    assert any(field in model.model_fields for model in models), field


def test_colormap_mode_has_no_control_in_either_toolkit():
    """It is ``frozen=True``, so a control wired to it could only raise.

    Section 10.4 listed it; section 6.5.1 proposal 4 removed it.  Asserted so
    it cannot creep back in with the rest of the labels controls.
    """
    for module in (qt_visuals, anywidget_visuals):
        assert not any("ColormapMode" in name for name in module.__all__)


# ---------------------------------------------------------------------------
# Layer 2: each control type round-trips a value
# ---------------------------------------------------------------------------


def _round_trip_cases(toolkit_getter):
    """(name, widget, edit, expected, foreign, expected_after) per control type."""
    visual_id = uuid4()
    return [
        (
            "bounded_slider",
            toolkit_getter("OpacitySlider")(visual_id, initial_value=1.0),
            0.4,
            0.25,
        ),
        (
            "float_spin",
            toolkit_getter("SizeSpin")(visual_id, initial_value=5.0),
            12.5,
            3.0,
        ),
        (
            "int_spin",
            toolkit_getter("SaltSpin")(visual_id, initial_value=0),
            42,
            7,
        ),
        (
            "choice",
            toolkit_getter("SideCombo")(visual_id, initial_value="both"),
            "front",
            "back",
        ),
        (
            "color_picker",
            toolkit_getter("UniformColorPicker")(
                visual_id, initial_value=(1.0, 1.0, 1.0, 1.0)
            ),
            (0.2, 0.4, 0.6, 0.8),
            (0.1, 0.1, 0.1, 0.5),
        ),
    ]


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_every_control_type_emits_the_edited_value(qtbot, toolkit):
    getter = _qt if toolkit == "qt" else _any

    for name, widget, edited, _foreign in _round_trip_cases(getter):
        emitted: list = []
        widget.changed.connect(emitted.append)

        if toolkit == "qt":
            _drive_qt(widget, name, edited)
        else:
            widget.value = widget._coerce(edited)

        assert len(emitted) == 1, f"{toolkit}/{name}: {emitted}"
        assert emitted[0].field == widget.field
        if name == "color_picker":
            assert tuple(emitted[0].value) == pytest.approx(edited)
        else:
            assert emitted[0].value == pytest.approx(edited)


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_every_control_type_applies_a_foreign_write_without_re_emitting(qtbot, toolkit):
    """The echo guard, per control type rather than assumed from the toggle.

    Section 6.1.1's negative control showed the guard is *defensive* on a plain
    ``<input>``; it becomes load-bearing for any control that re-dispatches on
    programmatic set, so it is checked here for each type rather than
    generalised from one.
    """
    getter = _qt if toolkit == "qt" else _any

    for name, widget, _edited, foreign in _round_trip_cases(getter):
        emitted: list = []
        widget.changed.connect(emitted.append)

        widget._apply(foreign)

        assert emitted == [], f"{toolkit}/{name} re-emitted on an inbound write"
        actual = widget.value() if toolkit == "qt" else widget.value
        if name == "color_picker":
            assert tuple(actual) == pytest.approx(foreign)
        else:
            assert actual == pytest.approx(foreign)


def _drive_qt(widget, name, value):
    """Perform a user-level edit on a Qt control of the given type."""
    if name == "bounded_slider":
        widget._control.setValue(value)
    elif name == "float_spin":
        widget._control.setValue(value)
    elif name == "int_spin":
        widget._spin.setValue(value)
    elif name == "choice":
        widget._control.setCurrentText(value)
    elif name == "color_picker":
        # The colour dialog is modal, so drive the two halves the dialog and
        # the alpha slider write, rather than opening it.
        widget._rgba = tuple(value)
        widget._emit(widget._rgba)
    else:  # pragma: no cover - the table above is closed
        raise AssertionError(name)


# ---------------------------------------------------------------------------
# Layer 2: the parts that are not just a value round trip
# ---------------------------------------------------------------------------


def test_the_opacity_range_is_read_off_the_model(qtbot):
    """Not restated: a tightened bound on the model tightens the slider."""
    from cellier.visuals._base_visual import BaseAppearance

    expected = field_bounds(BaseAppearance, "opacity")
    assert expected == (0.0, 1.0)
    assert _qt("OpacitySlider")._default_range == expected
    assert _any("OpacitySlider")._default_range == expected


def test_a_float_spin_range_is_overridable_at_construction(qtbot):
    """The escape hatch for the ``world``-space size/thickness fields."""
    visual_id = uuid4()

    qt_widget = _qt("SizeSpin")(visual_id, initial_value=5.0, value_range=(0.0, 5000.0))
    any_widget = _any("SizeSpin")(
        visual_id, initial_value=5.0, value_range=(0.0, 5000.0)
    )

    assert qt_widget._control.maximum() == pytest.approx(5000.0)
    assert any_widget.max == pytest.approx(5000.0)
    # And the default still applies when the keyword is omitted.
    assert _any("SizeSpin")(visual_id)._default_range == (0.1, 100.0)


def test_choices_come_from_the_models_literal_not_a_restated_list(qtbot):
    """In-memory and multiscale labels offer different render modes."""
    from cellier.visuals._label_memory import InMemoryLabelsAppearance
    from cellier.visuals._labels import MultiscaleLabelsAppearance

    in_memory = literal_choices(InMemoryLabelsAppearance(), "render_mode")
    multiscale = literal_choices(MultiscaleLabelsAppearance(), "render_mode")

    assert in_memory == ("iso_categorical", "flat_categorical")
    assert multiscale == ("iso_categorical", "flat_categorical", "smooth_iso")

    visual_id = uuid4()
    widget = _qt("LabelsRenderModeCombo")(visual_id, choices=multiscale)
    assert widget.choices == multiscale
    assert _any("LabelsRenderModeCombo")(visual_id, choices=multiscale).choices == list(
        multiscale
    )


def test_the_debug_render_mode_never_reaches_a_control():
    """``gradient_debug`` renders a diagnostic that looks like a bug.

    Denylisted rather than allowlisted, so a future *real* mode appears
    automatically (section 6.5.1 proposal 4).
    """
    from cellier.visuals._labels import MultiscaleLabelsAppearance

    assert "gradient_debug" not in literal_choices(
        MultiscaleLabelsAppearance(), "render_mode"
    )
    # ...and it really is on the model, so this is a filter and not a typo.
    assert (
        "gradient_debug"
        in MultiscaleLabelsAppearance.model_fields["render_mode"].annotation.__args__
    )


def test_a_choice_ignores_a_value_it_does_not_offer(qtbot):
    """Better a stale control than a blanked one."""
    visual_id = uuid4()
    for widget, read in (
        (_qt("SideCombo")(visual_id, initial_value="both"), lambda w: w.value()),
        (_any("SideCombo")(visual_id, initial_value="both"), lambda w: w.value),
    ):
        widget._apply("not_an_option")
        assert read(widget) in ("both", "not_an_option")

    qt_widget = _qt("SideCombo")(visual_id, initial_value="both")
    qt_widget._apply("not_an_option")
    assert qt_widget.value() == "both"


def test_only_the_salt_spin_offers_a_shuffle_button(qtbot):
    """The button is a trait, so one int-spin asset serves both fields."""
    assert _qt("SaltSpin")._shuffle is True
    assert _any("SaltSpin")._shuffle is True
    assert _qt("BackgroundLabelSpin")._shuffle is False
    assert _any("BackgroundLabelSpin")._shuffle is False

    visual_id = uuid4()
    assert _any("SaltSpin")(visual_id).shuffle is True
    assert _any("BackgroundLabelSpin")(visual_id).shuffle is False


def test_shuffle_writes_a_new_value_in_range(qtbot):
    widget = _qt("SaltSpin")(uuid4(), initial_value=0)
    emitted: list = []
    widget.changed.connect(emitted.append)

    widget._on_shuffle()

    assert len(emitted) == 1
    low, high = widget._range
    assert low <= emitted[0].value <= high


def test_the_background_label_range_spans_negative_ids(qtbot):
    """Label stores are ``int32`` and nothing forbids a negative id."""
    low, high = _qt("BackgroundLabelSpin")._default_range
    assert low < 0 < high
    assert _any("BackgroundLabelSpin")._default_range == (low, high)


# ---------------------------------------------------------------------------
# The RGBA/hex conversion the colour control carries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rgba", "expected"),
    [
        ((0.0, 0.0, 0.0, 1.0), "#000000"),
        ((1.0, 1.0, 1.0, 1.0), "#ffffff"),
        ((1.0, 0.0, 0.0, 0.5), "#ff0000"),
    ],
)
def test_rgba_to_hex_drops_alpha(rgba, expected):
    """Colour inputs are RGB-only in both toolkits; alpha has its own control."""
    assert rgba_to_hex(rgba) == expected


def test_hex_to_rgba_round_trips_through_the_swatch():
    original = (0.2, 0.4, 0.6, 0.8)
    recovered = hex_to_rgba(rgba_to_hex(original), original[3])
    assert recovered == pytest.approx(original, abs=1 / 255)


def test_rgba_to_hex_clamps_out_of_range_components():
    assert rgba_to_hex((-1.0, 2.0, 0.5, 1.0)) == "#00ff80"


def test_as_rgba_fills_a_missing_alpha():
    assert as_rgba([0.1, 0.2, 0.3]) == (0.1, 0.2, 0.3, 1.0)
    assert as_rgba((0.1, 0.2, 0.3, 0.4)) == (0.1, 0.2, 0.3, 0.4)


def test_a_colour_edit_emits_one_event_carrying_all_four_components(qtbot):
    """Not one event per component: the whole RGBA travels together."""
    for widget, read in (
        (_qt("UniformColorPicker")(uuid4()), lambda w: w.value()),
        (_any("UniformColorPicker")(uuid4()), lambda w: w.value),
    ):
        emitted: list = []
        widget.changed.connect(emitted.append)

        widget._apply((0.1, 0.2, 0.3, 1.0))
        assert emitted == []

        if hasattr(widget, "_on_alpha_changed"):
            widget._on_alpha_changed(0.25)
        else:
            widget.value = [0.1, 0.2, 0.3, 0.25]

        assert len(emitted) == 1
        assert tuple(emitted[0].value) == pytest.approx((0.1, 0.2, 0.3, 0.25))
        assert tuple(read(widget)) == pytest.approx((0.1, 0.2, 0.3, 0.25))


# ---------------------------------------------------------------------------
# One asset per control type (design section 10.2 / 6.2.1)
# ---------------------------------------------------------------------------


def test_each_control_type_has_exactly_one_shared_esm_asset():
    """Every field class of a type points at the same ``FileContents``.

    ``AnyWidget.__init_subclass__`` only coerces ``_esm`` for a class that
    declares it in its **own** ``__dict__``, so layer-3 classes inherit the
    layer-2 object rather than re-reading the file.  That is what makes "one
    ``.js`` per control type, not per field" true rather than aspirational --
    and it is why the section 6.1.1 browser procedure runs once per asset
    (section 6.5.2 decision 8).
    """
    from cellier.gui.anywidget.visuals._base import (
        AnywidgetBoundedSlider,
        AnywidgetChoice,
        AnywidgetColorPicker,
        AnywidgetFloatSpin,
        AnywidgetIntSpin,
        AnywidgetToggle,
    )

    by_type = {
        AnywidgetToggle: ["VisibleToggle", "WireframeToggle", "NodeVisibleToggle"],
        AnywidgetBoundedSlider: ["OpacitySlider"],
        AnywidgetFloatSpin: ["SizeSpin", "ShininessSpin", "EdgeThicknessSpin"],
        AnywidgetIntSpin: ["SaltSpin", "BackgroundLabelSpin"],
        AnywidgetChoice: ["SideCombo", "SizeSpaceCombo", "LabelsRenderModeCombo"],
        AnywidgetColorPicker: ["UniformColorPicker", "NodeColorPicker"],
    }

    for base, stems in by_type.items():
        for stem in stems:
            field_class = _any(stem)
            assert issubclass(field_class, base), stem
            assert field_class._esm is base._esm, stem
            assert field_class._css is base._css, stem

    # ...and no two control types share an asset, or a change to one would
    # silently change another.
    assets = [base._esm for base in by_type]
    assert len({id(asset) for asset in assets}) == len(assets)


def test_the_js_reads_generic_traits_not_field_names():
    """Extends the stage-0 assertion to the five new assets.

    A control type whose JS named a field trait would need one asset per
    field, which is the duplication the three-layer design exists to remove.
    """
    from pathlib import Path

    import cellier.gui.anywidget.visuals._base as base_module

    static = Path(base_module.__file__).parent / "static"
    field_names = {field for _stem, field, _label in CATALOG}

    for asset in (
        "toggle.js",
        "bounded_slider.js",
        "float_spin.js",
        "int_spin.js",
        "choice.js",
        "color_picker.js",
    ):
        source = (static / asset).read_text()
        for field in field_names:
            assert f'model.get("{field}")' not in source, f"{asset} names {field}"
            assert f'model.set("{field}"' not in source, f"{asset} names {field}"
        assert 'model.get("value")' in source, asset
        assert 'model.get("label")' in source, asset
