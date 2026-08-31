"""``controls=`` for the non-image visual types.

Stage 5 of ``plans/convenience_cleanup.md`` (section 11): the API half of
section 8.4.  Stage 4 built the widgets; this makes them reachable by giving
every visual family a config class and every ``add_*`` method a ``controls=``
argument.

The highest-value tests here are the end-to-end ones (section 11.5): a config
with ``appearance=True`` produces a dock containing the expected controls, per
visual type.  That is what actually proves stages 3, 4 and 5 line up -- the
vocabulary, the widgets, and the plumbing are three separate declarations and
nothing else checks that they agree.
"""

from __future__ import annotations

import pytest

from cellier.convenience import OrthoViewer, Viewer
from cellier.convenience.gui._controls_config import (
    GraphControlsConfig,
    LabelsControlsConfig,
    LinesControlsConfig,
    MeshControlsConfig,
    MultiscaleLabelsControlsConfig,
    PointsControlsConfig,
)
from cellier.convenience.layout._shared import (
    appearance_specs,
    select_appearance_target,
)
from cellier.visuals._mesh_memory import MeshFlatAppearance, MeshPhongAppearance

_PANELS = ("xy", "xz", "yz", "vol")


def _add(viewer, kind, stores, **kwargs):
    """Add one visual of *kind* to *viewer*, forwarding ``controls=``."""
    if kind == "labels":
        return viewer.add_labels(stores["labels"], **kwargs)
    if kind == "mesh":
        return viewer.add_mesh(
            stores["mesh"], appearance=MeshFlatAppearance(), **kwargs
        )
    if kind == "points":
        return viewer.add_points(stores["points"], **kwargs)
    if kind == "lines":
        return viewer.add_lines(stores["lines"], **kwargs)
    if kind == "graph":
        return viewer.add_graph(stores["graph"], **kwargs)
    raise AssertionError(kind)


CONFIGS = {
    "labels": LabelsControlsConfig,
    "mesh": MeshControlsConfig,
    "points": PointsControlsConfig,
    "lines": LinesControlsConfig,
    "graph": GraphControlsConfig,
}


@pytest.fixture
def stores(labels_store, mesh_store, points_store, lines_store, graph_store):
    return {
        "labels": labels_store,
        "mesh": mesh_store,
        "points": points_store,
        "lines": lines_store,
        "graph": graph_store,
    }


# ---------------------------------------------------------------------------
# The twelve new signatures record the config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_viewer_add_records_the_config(kind, stores):
    viewer = Viewer(("z", "y", "x"))
    config = CONFIGS[kind](appearance=True)

    visual = _add(viewer, kind, stores, controls=config)

    assert viewer._controls_configs == {visual.id: config}


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_viewer_add_without_controls_records_nothing(kind, stores):
    viewer = Viewer(("z", "y", "x"))
    _add(viewer, kind, stores)
    assert viewer._controls_configs == {}


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_ortho_add_records_the_config_and_the_panel_group(kind, stores):
    ortho = OrthoViewer(("z", "y", "x"))
    config = CONFIGS[kind](appearance=True)

    visuals = _add(ortho, kind, stores, controls=config)

    panel_ids = [visuals[key].id for key in _PANELS]
    assert ortho._controls_configs == {panel_ids[0]: config}
    assert ortho._visual_groups[panel_ids[0]] == panel_ids


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_ortho_add_without_controls_records_nothing(kind, stores):
    ortho = OrthoViewer(("z", "y", "x"))
    _add(ortho, kind, stores)
    assert ortho._controls_configs == {}
    assert ortho._visual_groups == {}


def test_add_labels_multiscale_takes_the_multiscale_config(multiscale_labels_store):
    from cellier.visuals._labels import MultiscaleLabelsAppearance

    viewer = Viewer(("z", "y", "x"))
    config = MultiscaleLabelsControlsConfig(appearance=["lod_bias"])

    visual = viewer.add_labels_multiscale(
        multiscale_labels_store,
        appearance=MultiscaleLabelsAppearance(),
        controls=config,
    )

    assert viewer._controls_configs == {visual.id: config}


# ---------------------------------------------------------------------------
# End to end: a default panel per visual type (section 11.5's key test)
# ---------------------------------------------------------------------------

EXPECTED_DEFAULT_TITLES = {
    "labels": [
        "Visible",
        "Opacity",
        "Render mode",
        "Salt",
        "Background label",
        "Bounding box",
    ],
    "mesh": [
        "Visible",
        "Opacity",
        "Color",
        "Side",
        "Wireframe",
        "Wireframe thickness",
        "Bounding box",
    ],
    "points": ["Visible", "Opacity", "Color", "Size", "Size space", "Bounding box"],
    "lines": [
        "Visible",
        "Opacity",
        "Color",
        "Thickness",
        "Thickness space",
        "Bounding box",
    ],
    "graph": [
        "Visible",
        "Opacity",
        "Nodes visible",
        "Node color",
        "Node size",
        "Node size space",
        "Edges visible",
        "Edge color",
        "Edge thickness",
        "Edge thickness space",
        "Bounding box",
    ],
}


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_appearance_true_builds_the_default_qt_dock(qtbot, kind, stores):
    """The test section 11.5 calls the highest-value one in the group.

    It fails if the vocabulary, the widget catalog and the plumbing disagree
    anywhere -- a valid field with no widget, a widget with no config entry, a
    config the renderer cannot resolve.
    """
    from cellier.convenience.layout._qt_renderer import _render_appearance_controls_qt
    from tests.convenience._qt_acceptance import assert_panel_renders, control_labels

    viewer = Viewer(("z", "y", "x"), gui="qt")
    _add(viewer, kind, stores, controls=CONFIGS[kind](appearance=True))

    container = _render_appearance_controls_qt(viewer)

    assert control_labels(container) == EXPECTED_DEFAULT_TITLES[kind]
    assert_panel_renders(container)


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_appearance_true_builds_the_same_anywidget_dock(kind, stores):
    """Same names, same order, other toolkit -- section 4.2's whole point."""
    from cellier.convenience.gui._appearance_widgets import (
        build_appearance_widgets_anywidget,
    )
    from tests.convenience._qt_acceptance import control_labels_anywidget

    viewer = Viewer(("z", "y", "x"), gui="anywidget")
    _add(viewer, kind, stores, controls=CONFIGS[kind](appearance=True))
    target = select_appearance_target(viewer)

    built = build_appearance_widgets_anywidget(
        target.visual, target.config, viewer.controller, target.visual_ids
    )

    assert control_labels_anywidget(built) == EXPECTED_DEFAULT_TITLES[kind]


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_appearance_false_still_hides_the_panel(qtbot, kind, stores):
    from cellier.convenience.layout._qt_renderer import _render_appearance_controls_qt

    viewer = Viewer(("z", "y", "x"), gui="qt")
    _add(viewer, kind, stores, controls=CONFIGS[kind](appearance=False))

    assert _render_appearance_controls_qt(viewer) is None


def test_a_phong_mesh_gets_its_own_fields_from_the_same_config(qtbot, mesh_store):
    """One ``MeshControlsConfig`` covers both mesh models.

    ``wireframe`` is flat-only and ``shininess`` phong-only; the panel follows
    the model the visual actually carries, so ``appearance=True`` is safe on
    either without two config classes.
    """
    from cellier.convenience.layout._qt_renderer import _render_appearance_controls_qt
    from tests.convenience._qt_acceptance import control_labels

    viewer = Viewer(("z", "y", "x"), gui="qt")
    # A phong mesh warns about scene lighting, which is unrelated to controls.
    with pytest.warns(UserWarning, match="requires lights"):
        viewer.add_mesh(
            mesh_store,
            appearance=MeshPhongAppearance(),
            controls=MeshControlsConfig(appearance=True),
        )

    assert control_labels(_render_appearance_controls_qt(viewer)) == [
        "Visible",
        "Opacity",
        "Color",
        "Side",
        "Shininess",
        "Flat shading",
        "Bounding box",
    ]


def test_the_labels_combo_offers_the_models_own_render_modes(qtbot, labels_store):
    """In-memory labels: two modes, and no image render modes."""
    from cellier.gui.qt.visuals import QtLabelsRenderModeCombo

    viewer = Viewer(("z", "y", "x"), gui="qt")
    visual = viewer.add_labels(
        labels_store, controls=LabelsControlsConfig(appearance=["render_mode"])
    )

    spec = appearance_specs(visual, viewer._controls_configs[visual.id]).specs[0]
    widget = QtLabelsRenderModeCombo(visual.id, **spec.values)

    assert widget.choices == ("iso_categorical", "flat_categorical")
    assert "mip" not in widget.choices


# ---------------------------------------------------------------------------
# Editing a rendered control reaches the model
# ---------------------------------------------------------------------------


def test_a_rendered_control_writes_the_model(qtbot, points_store):
    from cellier.convenience.layout._qt_renderer import _render_dock_qt
    from cellier.convenience.layout._spec import AppearanceControls

    viewer = Viewer(("z", "y", "x"), gui="qt")
    visual = viewer.add_points(
        points_store, controls=PointsControlsConfig(appearance=["size"])
    )

    container = _render_dock_qt(AppearanceControls(), viewer)
    from PySide6.QtWidgets import QDoubleSpinBox

    container.findChild(QDoubleSpinBox).setValue(12.0)

    assert visual.appearance.size == pytest.approx(12.0)


def test_an_ortho_non_image_edit_reaches_all_four_panels(qtbot, points_store):
    """The stage-2 fan-out, now for a visual type stage 2 did not cover."""
    from cellier.convenience.layout._qt_renderer import _render_dock_qt
    from cellier.convenience.layout._spec import AppearanceControls

    ortho = OrthoViewer(("z", "y", "x"))
    visuals = ortho.add_points(
        points_store, controls=PointsControlsConfig(appearance=["size"])
    )

    container = _render_dock_qt(AppearanceControls(), ortho)
    from PySide6.QtWidgets import QDoubleSpinBox

    container.findChild(QDoubleSpinBox).setValue(9.5)

    for key in _PANELS:
        assert visuals[key].appearance.size == pytest.approx(9.5)


def test_a_visible_toggle_reaches_the_model_on_its_own_event(qtbot, mesh_store):
    """``visible`` travels on ``VisualVisibilityChangedEvent`` (section 6.4.1).

    It is the one field routed specially, and now the first row of every
    default panel, so it is worth an end-to-end check rather than only the
    widget-level one.
    """
    from cellier.convenience.layout._qt_renderer import _render_dock_qt
    from cellier.convenience.layout._spec import AppearanceControls

    viewer = Viewer(("z", "y", "x"), gui="qt")
    visual = viewer.add_mesh(
        mesh_store,
        appearance=MeshFlatAppearance(),
        controls=MeshControlsConfig(appearance=["visible"]),
    )
    assert visual.appearance.visible is True

    container = _render_dock_qt(AppearanceControls(), viewer)
    from PySide6.QtWidgets import QCheckBox

    container.findChild(QCheckBox).setChecked(False)

    assert visual.appearance.visible is False


# ---------------------------------------------------------------------------
# Vocabulary, per new config class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_every_new_config_accepts_the_universal_fields(kind):
    assert CONFIGS[kind](appearance=["visible", "opacity"]).appearance == [
        "visible",
        "opacity",
    ]


@pytest.mark.parametrize(
    ("kind", "field"),
    [
        ("labels", "size"),
        ("mesh", "thickness"),
        ("points", "wireframe"),
        ("lines", "node_size"),
        ("graph", "shininess"),
    ],
)
def test_a_field_from_another_visual_family_is_rejected(kind, field):
    """The vocabulary is per family, so a mesh field is not a points field."""
    with pytest.raises(ValueError, match=field):
        CONFIGS[kind](appearance=[field])


def test_lod_bias_is_multiscale_labels_only():
    with pytest.raises(ValueError, match="lod_bias"):
        LabelsControlsConfig(appearance=["lod_bias"])
    assert MultiscaleLabelsControlsConfig(appearance=["lod_bias"]).appearance == [
        "lod_bias"
    ]


def test_colormap_mode_is_not_in_the_labels_vocabulary():
    """It is ``frozen=True``; a control for it could only raise."""
    with pytest.raises(ValueError, match="colormap_mode"):
        LabelsControlsConfig(appearance=["colormap_mode"])


def test_the_shared_widget_table_agrees_with_the_field_classes():
    """One name per control, not two that happen to match.

    ``APPEARANCE_FIELD_WIDGETS`` names each single-field control for the
    toolkit-neutral spec layer, and each widget class names itself with
    ``_label`` -- and since ``plans/label_ownership_unification.md`` the class
    is what actually gets drawn, in both toolkits.  Letting the two drift would
    mean a panel whose controls are named one thing and asserted to be named
    another.
    """
    from cellier.gui._appearance_fields import (
        APPEARANCE_FIELD_WIDGETS,
        field_widget_class,
    )

    for kind, (_stem, title) in APPEARANCE_FIELD_WIDGETS.items():
        for toolkit in ("qt", "anywidget"):
            widget_class = field_widget_class(kind, toolkit)
            assert widget_class._label == title, (
                f"{widget_class.__name__}._label != "
                f"APPEARANCE_FIELD_WIDGETS[{kind!r}] title"
            )


def test_composite_default_titles_match_the_shared_vocabulary():
    """The same check for the controls that read several fields at once.

    A composite's ``DEFAULT_TITLE`` is what it calls itself when constructed
    directly; ``_CONTROL_TITLES`` is what the renderers pass it.  They are two
    declarations of one name, so they are pinned together.
    """
    from cellier.convenience.layout._shared import _CONTROL_TITLES
    from cellier.gui.anywidget import AnywidgetDatasetInfo
    from cellier.gui.anywidget.visuals import (
        AnywidgetAABBWidget,
        AnywidgetClimSlider,
        AnywidgetColormapControl,
        AnywidgetLodBiasSlider,
        AnywidgetVolumeRenderControls,
    )
    from cellier.gui.qt.visuals import (
        QtAABBWidget,
        QtClimRangeSlider,
        QtColormapComboBox,
        QtLodBiasSlider,
        QtVolumeRenderControls,
    )

    # ``dataset_info`` has no Qt widget -- a documented gap (design section
    # 7.1), not an omission here.
    composites = {
        "color_map": (QtColormapComboBox, AnywidgetColormapControl),
        "clim": (QtClimRangeSlider, AnywidgetClimSlider),
        "render": (QtVolumeRenderControls, AnywidgetVolumeRenderControls),
        "lod_bias": (QtLodBiasSlider, AnywidgetLodBiasSlider),
        "aabb": (QtAABBWidget, AnywidgetAABBWidget),
        "dataset_info": (None, AnywidgetDatasetInfo),
    }
    assert set(composites) == set(_CONTROL_TITLES)

    for kind, classes in composites.items():
        for widget_class in classes:
            if widget_class is None:
                continue
            assert widget_class.DEFAULT_TITLE == _CONTROL_TITLES[kind], (
                f"{widget_class.__name__}.DEFAULT_TITLE != _CONTROL_TITLES[{kind!r}]"
            )


def test_every_valid_field_name_has_a_widget():
    """Section 10.1's rule, checked across every config class at once.

    A name that passes validation and then renders nothing is the exact bug
    stage 3 removed; widening a vocabulary past the widgets would reintroduce
    it.
    """
    from cellier.convenience.gui import _controls_config
    from cellier.gui._appearance_fields import APPEARANCE_FIELD_WIDGETS

    bespoke = {"color_map", "clim", "render", "lod_bias"}
    config_classes = [
        value
        for value in vars(_controls_config).values()
        if isinstance(value, type)
        and issubclass(value, _controls_config.BaseControlsConfig)
    ]
    assert config_classes

    for config_class in config_classes:
        for field, kind in config_class.APPEARANCE_CONTROLS.items():
            assert kind in APPEARANCE_FIELD_WIDGETS or kind in bespoke, (
                f"{config_class.__name__}.{field} -> {kind} has no widget"
            )
