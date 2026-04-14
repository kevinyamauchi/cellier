"""Tests for SceneManager.swap_node and _active_nodes tracking."""

from unittest.mock import MagicMock
from uuid import uuid4

import pygfx as gfx
import pytest

from cellier.v2.render.scene_manager import SceneManager


@pytest.fixture
def scene_manager():
    return SceneManager(scene_id=uuid4())


def _make_mock_visual(node_a, node_b=None):
    """Build a mock GFX visual with get_node_for_dims returning node_a for 3D
    and node_b (or node_a if None) for 2D."""
    if node_b is None:
        node_b = node_a
    visual = MagicMock()
    visual.visual_model_id = uuid4()
    visual.get_node_for_dims = lambda axes: node_a if len(axes) == 3 else node_b
    return visual


def test_swap_node_different_nodes(scene_manager):
    """Swapping between two distinct nodes removes old and adds new."""
    node_a = gfx.Group()
    node_b = gfx.Group()
    visual_id = uuid4()

    scene_manager._scene.add(node_a)
    scene_manager._active_nodes[visual_id] = node_a

    scene_manager.swap_node(visual_id, node_b)

    assert node_a.parent is None  # removed
    assert node_b.parent is not None  # added
    assert scene_manager.get_active_node(visual_id) is node_b


def test_swap_node_same_node_is_noop(scene_manager):
    """swap_node with the same node leaves the scene graph untouched."""
    node = gfx.Group()
    visual_id = uuid4()

    scene_manager._scene.add(node)
    scene_manager._active_nodes[visual_id] = node

    scene_manager.swap_node(visual_id, node)  # same object

    assert node.parent is not None  # still in scene
    assert scene_manager.get_active_node(visual_id) is node


def test_swap_node_new_none_removes_old(scene_manager):
    """swap_node(visual_id, None) removes the current node."""
    node = gfx.Group()
    visual_id = uuid4()

    scene_manager._scene.add(node)
    scene_manager._active_nodes[visual_id] = node

    scene_manager.swap_node(visual_id, None)

    assert node.parent is None
    assert scene_manager.get_active_node(visual_id) is None


def test_add_visual_populates_active_nodes(scene_manager):
    """add_visual stores the initial node in _active_nodes."""
    node_3d = gfx.Group()
    node_2d = gfx.Group()
    visual = _make_mock_visual(node_3d, node_2d)

    scene_manager.add_visual(visual, displayed_axes=(0, 1, 2))

    assert scene_manager.get_active_node(visual.visual_model_id) is node_3d
    assert node_3d.parent is not None


def test_add_visual_selects_2d_node(scene_manager):
    node_3d = gfx.Group()
    node_2d = gfx.Group()
    visual = _make_mock_visual(node_3d, node_2d)

    scene_manager.add_visual(visual, displayed_axes=(1, 2))

    assert scene_manager.get_active_node(visual.visual_model_id) is node_2d
    assert node_2d.parent is not None
    assert node_3d.parent is None


def test_add_visual_raises_when_get_node_returns_none(scene_manager):
    """add_visual raises ValueError if get_node_for_dims returns None."""
    visual = MagicMock()
    visual.visual_model_id = uuid4()
    visual.get_node_for_dims = lambda axes: None

    with pytest.raises(ValueError, match="returned None"):
        scene_manager.add_visual(visual, displayed_axes=(0, 1, 2))
