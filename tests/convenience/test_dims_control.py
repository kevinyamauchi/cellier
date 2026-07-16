"""Tests for ``cellier.convenience.Viewer.set_displayed_dimensions``.

Covers the two validation guards and the 2D <-> 3D roundtrip that saves and
restores a slice position when an axis cycles from displayed back to sliced.
"""

from __future__ import annotations

import pytest

from cellier.convenience import Viewer


def test_rejects_wrong_arity():
    viewer = Viewer(("z", "y", "x"))
    with pytest.raises(ValueError, match="2 or 3 entries"):
        viewer.set_displayed_dimensions(("x",))
    with pytest.raises(ValueError, match="2 or 3 entries"):
        viewer.set_displayed_dimensions(("t", "z", "y", "x"))


def test_rejects_unknown_axis_name():
    viewer = Viewer(("z", "y", "x"))
    with pytest.raises(ValueError, match="Unknown axis names"):
        viewer.set_displayed_dimensions(("q", "x"))


def test_switch_2d_to_3d_sets_displayed_axes():
    viewer = Viewer(("t", "z", "y", "x"), dim="2d")
    scene = viewer.scene
    assert tuple(scene.dims.selection.displayed_axes) == (2, 3)

    viewer.set_displayed_dimensions(("z", "y", "x"))
    assert tuple(scene.dims.selection.displayed_axes) == (1, 2, 3)
    # z (axis 1) is now displayed, so it drops out of slice_indices.
    assert 1 not in scene.dims.selection.slice_indices


def test_roundtrip_restores_saved_slice_position():
    viewer = Viewer(("t", "z", "y", "x"), dim="2d")
    scene = viewer.scene

    # Seed a non-default slice position on z (axis 1) while it is sliced.
    slices = dict(scene.dims.selection.slice_indices)
    slices[1] = 5
    viewer.controller.update_slice_indices(scene.id, slices)
    assert scene.dims.selection.slice_indices[1] == 5

    # Expand to 3D: z becomes displayed and its position is saved.
    viewer.set_displayed_dimensions(("z", "y", "x"))
    assert 1 not in scene.dims.selection.slice_indices

    # Contract back to 2D: z is sliced again and restored to its saved value.
    viewer.set_displayed_dimensions(("y", "x"))
    assert tuple(scene.dims.selection.displayed_axes) == (2, 3)
    assert scene.dims.selection.slice_indices[1] == pytest.approx(5)


def test_unseen_axis_defaults_to_zero_on_contract():
    viewer = Viewer(("t", "z", "y", "x"), dim="3d")
    scene = viewer.scene
    # Displaying z, y, x leaves t (axis 0) sliced.
    viewer.set_displayed_dimensions(("z", "y", "x"))
    assert tuple(scene.dims.selection.displayed_axes) == (1, 2, 3)

    # Contract to y, x: z (axis 1) was never seeded, so it restores to 0.
    viewer.set_displayed_dimensions(("y", "x"))
    assert scene.dims.selection.slice_indices[1] == pytest.approx(0)
