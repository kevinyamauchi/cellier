"""Tests for the background API on the convenience viewers."""

import numpy as np

from cellier.convenience import OrthoViewer, Viewer
from cellier.scene import BackgroundAppearance


def _top_color(viewer_controller, scene_id) -> np.ndarray:
    material = viewer_controller._render_manager._scenes[scene_id]._background_material
    return np.asarray(material.color_top_left)


def test_viewer_background_property_is_the_scene_model():
    viewer = Viewer(axis_labels=("y", "x"), dim="2d")
    assert viewer.background is viewer.scene.background


def test_viewer_background_field_change_reaches_the_render_layer():
    viewer = Viewer(axis_labels=("y", "x"), dim="2d")

    viewer.background.top_color = (1.0, 0.0, 0.0, 1.0)

    np.testing.assert_allclose(
        _top_color(viewer.controller, viewer.scene.id),
        (1.0, 0.0, 0.0, 1.0),
        atol=1e-6,
    )


def test_viewer_background_can_be_replaced():
    viewer = Viewer(axis_labels=("y", "x"), dim="2d")

    viewer.background = BackgroundAppearance(mode="uniform", color=(0.0, 1.0, 0.0, 1.0))

    assert viewer.scene.background.mode == "uniform"
    np.testing.assert_allclose(
        _top_color(viewer.controller, viewer.scene.id),
        (0.0, 1.0, 0.0, 1.0),
        atol=1e-6,
    )
    # The new model stays connected.
    viewer.background.color = (0.0, 0.0, 1.0, 1.0)
    np.testing.assert_allclose(
        _top_color(viewer.controller, viewer.scene.id),
        (0.0, 0.0, 1.0, 1.0),
        atol=1e-6,
    )


def test_ortho_set_background_applies_to_all_panels():
    viewer = OrthoViewer(axis_labels=("z", "y", "x"))

    viewer.set_background(
        BackgroundAppearance(mode="uniform", color=(1.0, 0.0, 0.0, 1.0))
    )

    for scene in viewer.scenes.values():
        np.testing.assert_allclose(
            _top_color(viewer.controller, scene.id), (1.0, 0.0, 0.0, 1.0), atol=1e-6
        )


def test_ortho_panels_get_independent_copies():
    """Editing one panel's background afterwards leaves the others alone."""
    viewer = OrthoViewer(axis_labels=("z", "y", "x"))
    viewer.set_background(
        BackgroundAppearance(mode="uniform", color=(1.0, 0.0, 0.0, 1.0))
    )

    viewer.scenes["xy"].background.color = (0.0, 0.0, 1.0, 1.0)

    np.testing.assert_allclose(
        _top_color(viewer.controller, viewer.scenes["xy"].id),
        (0.0, 0.0, 1.0, 1.0),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _top_color(viewer.controller, viewer.scenes["vol"].id),
        (1.0, 0.0, 0.0, 1.0),
        atol=1e-6,
    )
