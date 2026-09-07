"""Tests for SceneManager background handling."""

from uuid import uuid4

import numpy as np
import pytest

from cellier.render.scene_manager import SceneManager
from cellier.scene._background import (
    DEFAULT_BOTTOM_COLOR,
    DEFAULT_TOP_COLOR,
    BackgroundAppearance,
)


def _corners(scene_manager: SceneManager) -> dict[str, np.ndarray]:
    material = scene_manager._background_material
    return {
        "bottom_left": np.asarray(material.color_bottom_left),
        "bottom_right": np.asarray(material.color_bottom_right),
        "top_left": np.asarray(material.color_top_left),
        "top_right": np.asarray(material.color_top_right),
    }


def test_default_background_matches_the_previous_hardcoded_colors():
    """The colors that used to be built inline in __init__ are the defaults."""
    scene_manager = SceneManager(scene_id=uuid4())
    corners = _corners(scene_manager)

    np.testing.assert_allclose(corners["bottom_left"], DEFAULT_BOTTOM_COLOR, atol=1e-6)
    np.testing.assert_allclose(corners["top_left"], DEFAULT_TOP_COLOR, atol=1e-6)
    # The gradient is vertical only: the two bottom corners agree, as do the
    # two top ones.
    np.testing.assert_allclose(corners["bottom_left"], corners["bottom_right"])
    np.testing.assert_allclose(corners["top_left"], corners["top_right"])


def test_background_is_in_the_scene_graph_and_retained():
    scene_manager = SceneManager(scene_id=uuid4())
    assert scene_manager.background.parent is scene_manager.scene


def test_construct_with_a_background():
    scene_manager = SceneManager(
        scene_id=uuid4(),
        background=BackgroundAppearance(mode="uniform", color=(1.0, 0.0, 0.0, 1.0)),
    )
    for color in _corners(scene_manager).values():
        np.testing.assert_allclose(color, (1.0, 0.0, 0.0, 1.0), atol=1e-6)


def test_set_background_uniform():
    scene_manager = SceneManager(scene_id=uuid4())
    scene_manager.set_background(
        BackgroundAppearance(mode="uniform", color=(0.0, 0.0, 1.0, 1.0))
    )
    for color in _corners(scene_manager).values():
        np.testing.assert_allclose(color, (0.0, 0.0, 1.0, 1.0), atol=1e-6)


def test_set_background_gradient():
    scene_manager = SceneManager(scene_id=uuid4())
    scene_manager.set_background(
        BackgroundAppearance(
            bottom_color=(0.0, 1.0, 0.0, 1.0), top_color=(0.0, 0.0, 0.0, 1.0)
        )
    )
    corners = _corners(scene_manager)
    np.testing.assert_allclose(corners["bottom_left"], (0.0, 1.0, 0.0, 1.0), atol=1e-6)
    np.testing.assert_allclose(corners["top_right"], (0.0, 0.0, 0.0, 1.0), atol=1e-6)


@pytest.mark.parametrize("visible", [True, False])
def test_set_background_visibility(visible):
    scene_manager = SceneManager(scene_id=uuid4())
    scene_manager.set_background(BackgroundAppearance(visible=visible))
    assert scene_manager.background.visible is visible


def test_set_background_reuses_the_same_object():
    """Colors are rewritten in place, so nothing is added to the scene graph."""
    scene_manager = SceneManager(scene_id=uuid4())
    background = scene_manager.background
    n_children = len(scene_manager.scene.children)

    scene_manager.set_background(BackgroundAppearance(mode="uniform"))

    assert scene_manager.background is background
    assert len(scene_manager.scene.children) == n_children
