"""Test the viewer model convenience functions."""

import pytest

from cellier.convenience import get_dims_with_canvas_id, get_dims_with_visual_id


def test_get_get_dims_with_visual_id(viewer_model_2d_points):
    """Test getting the dims model from a scene."""
    viewer_model = viewer_model_2d_points["viewer"]
    visual_model = viewer_model_2d_points["visual"]

    dims = get_dims_with_visual_id(viewer_model, visual_model.id)

    assert dims is next(iter(viewer_model.scenes.scenes.values())).dims


def test_get_get_dims_with_visual_id_bad_id(viewer_model_2d_points):
    """Test that attempting to get the dims with a bad ID raises error."""
    viewer_model = viewer_model_2d_points["viewer"]

    # check that the dims manager is None
    with pytest.raises(ValueError):
        get_dims_with_visual_id(viewer_model, "fake_id")


def test_get_get_dims_with_canvas_id(viewer_model_2d_points):
    """Test getting the dims model from a scene."""
    viewer_model = viewer_model_2d_points["viewer"]
    canvas_model = viewer_model_2d_points["canvas"]

    dims = get_dims_with_canvas_id(viewer_model, canvas_model.id)

    assert dims is next(iter(viewer_model.scenes.scenes.values())).dims


def test_get_get_dims_with_canvas_id_bad_id(viewer_model_2d_points):
    """Test that attempting to get the dims with a bad ID raises error."""
    viewer_model = viewer_model_2d_points["viewer"]

    # check that the dims manager is None
    with pytest.raises(ValueError):
        get_dims_with_canvas_id(viewer_model, "fake_id")
