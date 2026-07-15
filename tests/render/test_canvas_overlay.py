"""Tests for the render-layer canvas overlay ``GFXCenteredAxes2D`` (Phase 3).

The overlay owns its own ``gfx.Scene`` + ``ScreenCoordsCamera`` and rebuilds
its two axis-line segments from the live camera each frame, so it can be tested
directly with a pygfx camera -- no full controller needed.  Covers construction
(line + optional labels, initial visibility, per-axis colours), the pixel-space
anchor placement for every ``corner`` mode, the frame-rebuild fast path, the
visibility toggle, and the world->screen direction helper.
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx
import pytest

from cellier.render.visuals._canvas_overlay import (
    GFXCenteredAxes2D,
    _compute_screen_dir,
)
from cellier.visuals._canvas_overlay import CenteredAxes2D, CenteredAxes2DAppearance


def _camera() -> gfx.OrthographicCamera:
    """A simple camera looking down -Z with an identity-ish orientation."""
    cam = gfx.OrthographicCamera(100, 100)
    cam.show_rect(-50, 50, -50, 50)
    return cam


def _overlay(**model_kwargs) -> GFXCenteredAxes2D:
    model_kwargs.setdefault("name", "axes")
    model = CenteredAxes2D(**model_kwargs)
    return GFXCenteredAxes2D(model=model, camera=_camera())


def _anchor(overlay: GFXCenteredAxes2D) -> tuple[float, float]:
    """The shared origin of both segments (positions row 0)."""
    pos = overlay._line.geometry.positions.data
    return float(pos[0, 0]), float(pos[0, 1])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_builds_line_and_labels():
    overlay = _overlay()
    assert isinstance(overlay.overlay_scene, gfx.Scene)
    assert isinstance(overlay.overlay_camera, gfx.ScreenCoordsCamera)
    assert overlay._line in overlay.overlay_scene.children
    # show_labels defaults True -> both text labels exist.
    assert overlay._label_a is not None
    assert overlay._label_b is not None


def test_construction_without_labels():
    overlay = _overlay(appearance=CenteredAxes2DAppearance(show_labels=False))
    assert overlay._label_a is None
    assert overlay._label_b is None


def test_construction_respects_initial_visibility():
    overlay = _overlay(visible=False)
    assert overlay._line.visible is False
    assert overlay._label_a.visible is False
    assert overlay._label_b.visible is False


def test_construction_uses_per_axis_colours():
    appearance = CenteredAxes2DAppearance(
        axis_a_color=(1.0, 0.0, 0.0, 1.0),
        axis_b_color=(0.0, 0.0, 1.0, 1.0),
    )
    overlay = _overlay(appearance=appearance)
    colors = overlay._line.geometry.colors.data
    # Vertices 0,1 -> axis A; 2,3 -> axis B.
    np.testing.assert_allclose(colors[0], (1.0, 0.0, 0.0, 1.0))
    np.testing.assert_allclose(colors[2], (0.0, 0.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# Placement (on_frame -> _rebuild_geometry)
# ---------------------------------------------------------------------------


def test_center_anchor_is_canvas_middle():
    overlay = _overlay(appearance=CenteredAxes2DAppearance(corner="center"))
    overlay.on_frame(200.0, 100.0)
    assert _anchor(overlay) == (100.0, 50.0)


@pytest.mark.parametrize(
    ("corner", "expected"),
    [
        ("top_left", (20.0, 20.0)),
        ("top_right", (180.0, 20.0)),
        ("bottom_left", (20.0, 80.0)),
        ("bottom_right", (180.0, 80.0)),
    ],
)
def test_corner_anchor_placement(corner, expected):
    overlay = _overlay(
        appearance=CenteredAxes2DAppearance(
            corner=corner, corner_offset_px=(20.0, 20.0)
        )
    )
    overlay.on_frame(200.0, 100.0)
    assert _anchor(overlay) == expected


def test_on_frame_rebuilds_on_size_change_only():
    overlay = _overlay(appearance=CenteredAxes2DAppearance(corner="center"))
    overlay.on_frame(200.0, 100.0)
    assert _anchor(overlay) == (100.0, 50.0)

    # A different size re-anchors.
    overlay.on_frame(400.0, 400.0)
    assert _anchor(overlay) == (200.0, 200.0)


def test_on_frame_is_noop_when_unchanged():
    overlay = _overlay(appearance=CenteredAxes2DAppearance(corner="center"))
    overlay.on_frame(200.0, 100.0)
    before = overlay._line.geometry.positions.data.copy()

    # Same size + camera -> early return, geometry untouched.
    overlay.on_frame(200.0, 100.0)
    np.testing.assert_array_equal(overlay._line.geometry.positions.data, before)


def test_segment_tips_are_length_px_from_anchor():
    length = 60.0
    overlay = _overlay(
        appearance=CenteredAxes2DAppearance(corner="center", length_px=length)
    )
    overlay.on_frame(200.0, 200.0)
    pos = overlay._line.geometry.positions.data
    anchor = pos[0, :2]
    tip_a = pos[1, :2]
    tip_b = pos[3, :2]
    assert np.linalg.norm(tip_a - anchor) == pytest.approx(length, abs=1e-3)
    assert np.linalg.norm(tip_b - anchor) == pytest.approx(length, abs=1e-3)


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------


def test_set_visible_toggles_line_and_labels():
    overlay = _overlay()
    overlay.set_visible(False)
    assert overlay._line.visible is False
    assert overlay._label_a.visible is False
    assert overlay._label_b.visible is False

    overlay.set_visible(True)
    assert overlay._line.visible is True
    assert overlay._label_a.visible is True
    assert overlay._label_b.visible is True


def test_set_visible_without_labels_does_not_raise():
    overlay = _overlay(appearance=CenteredAxes2DAppearance(show_labels=False))
    overlay.set_visible(False)
    overlay.set_visible(True)
    assert overlay._line.visible is True


# ---------------------------------------------------------------------------
# world->screen direction helper
# ---------------------------------------------------------------------------


def test_compute_screen_dir_returns_unit_vector():
    result = _compute_screen_dir((0.0, 1.0, 0.0), _camera())
    assert result.shape == (2,)
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)


def test_compute_screen_dir_zero_projection_falls_back():
    """A direction parallel to the view axis projects to ~0 -> (1, 0) fallback."""
    cam = _camera()
    # OrthographicCamera looks down -Z; a pure +Z world direction has no screen
    # projection, triggering the near-zero fallback.
    result = _compute_screen_dir((0.0, 0.0, 1.0), cam)
    np.testing.assert_array_equal(result, np.array([1.0, 0.0], dtype=np.float32))
