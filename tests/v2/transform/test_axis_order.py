"""Tests for the axis-order primitives in cellier.v2.transform._axis_order."""

from __future__ import annotations

import numpy as np
import pytest

from cellier.v2.transform import AffineTransform
from cellier.v2.transform._axis_order import select_axes, swap_axes

# ---------------------------------------------------------------------------
# select_axes (sequence)
# ---------------------------------------------------------------------------


def test_select_axes_subset_preserves_order():
    values = (10, 20, 30, 40, 50)
    assert select_axes(values, (0, 4)) == (10, 50)
    assert select_axes(values, (4, 0)) == (50, 10)
    assert select_axes(values, (1, 3, 4)) == (20, 40, 50)


def test_select_axes_empty():
    assert select_axes((1, 2, 3), ()) == ()


def test_select_axes_full_is_identity_when_in_order():
    values = (1, 2, 3)
    assert select_axes(values, (0, 1, 2)) == values


# ---------------------------------------------------------------------------
# swap_axes (sequence)
# ---------------------------------------------------------------------------


def test_swap_axes_reverse():
    assert swap_axes((10, 20, 30), (2, 1, 0)) == (30, 20, 10)


def test_swap_axes_identity():
    assert swap_axes((10, 20, 30), (0, 1, 2)) == (10, 20, 30)


def test_swap_axes_arbitrary_permutation():
    assert swap_axes(("a", "b", "c", "d"), (3, 0, 2, 1)) == ("d", "a", "c", "b")


def test_swap_axes_rejects_non_permutation():
    # repeated index
    with pytest.raises(ValueError, match="permutation"):
        swap_axes((10, 20, 30), (0, 0, 1))
    # wrong length
    with pytest.raises(ValueError, match="permutation"):
        swap_axes((10, 20, 30), (0, 1))
    # out-of-range index
    with pytest.raises(ValueError, match="permutation"):
        swap_axes((10, 20, 30), (0, 1, 5))


# ---------------------------------------------------------------------------
# AffineTransform.swap_axes
# ---------------------------------------------------------------------------


def test_transform_swap_axes_reverse_3d():
    """A diagonal scale transform reversed by (2,1,0) reorders the diagonal."""
    t = AffineTransform.from_scale_and_translation((4.0, 2.0, 3.0), (10.0, 20.0, 30.0))
    swapped = t.swap_axes((2, 1, 0))
    assert swapped.ndim == 3
    # Diagonal: was (4, 2, 3), becomes (3, 2, 4)
    np.testing.assert_allclose(swapped.matrix[0, 0], 3.0)
    np.testing.assert_allclose(swapped.matrix[1, 1], 2.0)
    np.testing.assert_allclose(swapped.matrix[2, 2], 4.0)
    # Translation: was (10, 20, 30), becomes (30, 20, 10)
    np.testing.assert_allclose(swapped.matrix[0, 3], 30.0)
    np.testing.assert_allclose(swapped.matrix[1, 3], 20.0)
    np.testing.assert_allclose(swapped.matrix[2, 3], 10.0)


def test_transform_swap_axes_identity_returns_equivalent():
    t = AffineTransform.from_scale((4.0, 2.0, 3.0))
    swapped = t.swap_axes((0, 1, 2))
    np.testing.assert_array_equal(swapped.matrix, t.matrix)


def test_transform_swap_axes_rejects_non_permutation():
    t = AffineTransform.from_scale((1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="permutation"):
        t.swap_axes((0, 0, 1))
    with pytest.raises(ValueError, match="permutation"):
        t.swap_axes((0, 1))


def test_transform_select_then_swap_matches_select_in_swapped_order():
    """select_axes(perm) on a sub-transform == select_axes(perm-of-original)."""
    t = AffineTransform.from_scale_and_translation(
        (1.0, 4.0, 2.0, 3.0), (0.0, 10.0, 20.0, 30.0)
    )
    # Select displayed axes (1, 3) in data order, then swap to (3, 1).
    composed = t.select_axes((1, 3)).swap_axes((1, 0))
    # Equivalent: select directly in display order.
    direct = t.select_axes((3, 1))
    np.testing.assert_array_equal(composed.matrix, direct.matrix)


def test_transform_select_axes_preserves_offdiagonal_under_swap():
    """Off-diagonal entries (shear) survive a swap and end up in transposed slots."""
    m = np.eye(4, dtype=np.float32)
    m[0, 1] = 0.5
    m[1, 0] = 0.3
    t = AffineTransform(matrix=m)
    swapped = t.swap_axes((1, 0, 2))
    # The original (0, 1) entry now lives at (1, 0).
    np.testing.assert_allclose(swapped.matrix[1, 0], 0.5)
    np.testing.assert_allclose(swapped.matrix[0, 1], 0.3)
