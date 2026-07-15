"""Tests for the Qt ``QtLodBiasSlider`` LOD-bias-control widget."""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("superqt")

from cellier.controller import CellierController  # noqa: E402
from cellier.data.image._zarr_multiscale_store import (  # noqa: E402
    MultiscaleZarrDataStore,
)
from cellier.gui.qt.visuals._lod_bias import QtLodBiasSlider  # noqa: E402
from cellier.scene.dims import CoordinateSystem  # noqa: E402
from cellier.transform import AffineTransform  # noqa: E402
from cellier.visuals import MultiscaleImageAppearance  # noqa: E402


def _make_store(small_zarr_store, **kwargs) -> MultiscaleZarrDataStore:
    defaults = {
        "zarr_path": str(small_zarr_store),
        "scale_names": ["s0", "s1"],
        "level_transforms": [
            AffineTransform.identity(ndim=3),
            AffineTransform.from_scale_and_translation(
                (2.0, 2.0, 2.0), (0.5, 0.5, 0.5)
            ),
        ],
    }
    defaults.update(kwargs)
    return MultiscaleZarrDataStore(**defaults)


def _make_controller_with_visual(small_zarr_store, initial_lod_bias=1.0):
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = _make_store(small_zarr_store)
    appearance = MultiscaleImageAppearance(
        color_map="viridis", clim=(0.0, 1.0), lod_bias=initial_lod_bias
    )
    visual = controller.add_image_multiscale(
        data=store, scene_id=scene.id, appearance=appearance, name="vol"
    )
    return controller, visual


def test_instantiate_smoke(qtbot):
    slider = QtLodBiasSlider(visual_id=object(), initial_lod_bias=2.0)
    qtbot.addWidget(slider.widget)
    assert slider._slider.value() == pytest.approx(2.0)


def test_slider_release_reaches_model(qtbot, small_zarr_store):
    controller, visual = _make_controller_with_visual(small_zarr_store)
    slider = QtLodBiasSlider(visual_id=visual.id)
    qtbot.addWidget(slider.widget)
    controller.connect_widget(slider, subscription_specs=slider.subscription_specs())

    slider._slider.setValue(3.5)
    slider._on_slider_released()  # simulate the user releasing the handle

    assert visual.appearance.lod_bias == pytest.approx(3.5)


def test_model_push_updates_slider_without_reemit(qtbot, small_zarr_store):
    controller, visual = _make_controller_with_visual(small_zarr_store)
    slider = QtLodBiasSlider(visual_id=visual.id)
    qtbot.addWidget(slider.widget)
    controller.connect_widget(slider, subscription_specs=slider.subscription_specs())

    emitted = []
    slider.changed.connect(emitted.append)

    controller.update_appearance_field(visual.id, "lod_bias", 4.0)

    assert slider._slider.value() == pytest.approx(4.0)
    assert emitted == []


def test_unrelated_field_change_ignored(qtbot, small_zarr_store):
    controller, visual = _make_controller_with_visual(small_zarr_store)
    slider = QtLodBiasSlider(visual_id=visual.id, initial_lod_bias=1.0)
    qtbot.addWidget(slider.widget)
    controller.connect_widget(slider, subscription_specs=slider.subscription_specs())

    controller.update_appearance_field(visual.id, "clim", (0.0, 500.0))

    assert slider._slider.value() == pytest.approx(1.0)
