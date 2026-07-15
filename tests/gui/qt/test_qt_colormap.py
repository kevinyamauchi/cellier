"""Tests for the Qt ``QtColormapComboBox`` colormap-control widget."""

from __future__ import annotations

import numpy as np
import pytest
from cmap import Colormap

pytest.importorskip("qtpy")
pytest.importorskip("superqt")

from cellier.controller import CellierController  # noqa: E402
from cellier.data.image._image_memory_store import ImageMemoryStore  # noqa: E402
from cellier.gui.qt.visuals._colormap import QtColormapComboBox  # noqa: E402
from cellier.scene.dims import CoordinateSystem  # noqa: E402
from cellier.visuals._image_memory import InMemoryImageAppearance  # noqa: E402


def _make_controller_with_visual(color_map="viridis"):
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("y", "x"))
    scene = controller.add_scene(
        dim="2d", coordinate_system=cs, name="main", render_modes={"2d"}
    )
    data = np.zeros((16, 16), dtype=np.float32)
    store = ImageMemoryStore(data=data)
    appearance = InMemoryImageAppearance(color_map=color_map, clim=(0.0, 1.0))
    visual = controller.add_image(data=store, scene_id=scene.id, appearance=appearance)
    return controller, visual


def test_instantiate_smoke(qtbot):
    combo = QtColormapComboBox(visual_id=object(), initial_colormap="viridis")
    qtbot.addWidget(combo.widget)
    assert "viridis" in combo._combo.currentColormap().name


def test_combo_edit_reaches_model(qtbot):
    controller, visual = _make_controller_with_visual()
    combo = QtColormapComboBox(visual_id=visual.id, initial_colormap="viridis")
    qtbot.addWidget(combo.widget)
    controller.connect_widget(combo, subscription_specs=combo.subscription_specs())

    combo._on_combo_changed(Colormap("magma"))

    assert "magma" in visual.appearance.color_map.name


def test_model_push_updates_combo_without_reemit(qtbot):
    controller, visual = _make_controller_with_visual()
    combo = QtColormapComboBox(visual_id=visual.id, initial_colormap="viridis")
    qtbot.addWidget(combo.widget)
    controller.connect_widget(combo, subscription_specs=combo.subscription_specs())

    emitted = []
    combo.changed.connect(emitted.append)

    controller.update_appearance_field(visual.id, "color_map", Colormap("cividis"))

    assert "cividis" in combo._combo.currentColormap().name
    assert emitted == []


def test_unrelated_field_change_ignored(qtbot):
    controller, visual = _make_controller_with_visual(color_map="viridis")
    combo = QtColormapComboBox(visual_id=visual.id, initial_colormap="viridis")
    qtbot.addWidget(combo.widget)
    controller.connect_widget(combo, subscription_specs=combo.subscription_specs())

    controller.update_appearance_field(visual.id, "clim", (0.0, 500.0))

    assert "viridis" in combo._combo.currentColormap().name
