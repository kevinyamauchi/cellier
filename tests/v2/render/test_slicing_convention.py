"""Tests for the unified round-half-up slice-index -> voxel convention.

Covers the shared helper (:func:`round_world_to_voxel`) and the invariant that
the in-memory and multiscale planning paths snap a world slice position to the
*same* voxel index at level 0 -- the guard that did not previously exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.v2._state import AxisAlignedSelectionState
from cellier.v2.render.visuals._image import _build_axis_selections_multiscale
from cellier.v2.render.visuals._image_memory import _transform_slice_indices
from cellier.v2.render.visuals._slicing import round_world_to_voxel
from cellier.v2.transform import AffineTransform

# ── round_world_to_voxel ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, size, expected",
    [
        # Exact integers map to themselves.
        (0.0, 10, 0),
        (3.0, 10, 3),
        # Half-integer ties round toward +inf (round-half-up).
        (2.5, 10, 3),
        (0.5, 10, 1),
        (-0.5, 10, 0),  # -> 0 before clamp, stays 0
        # Below/above ties.
        (2.49, 10, 2),
        (2.51, 10, 3),
        # Floating-point jitter around a center.
        (2.4999999, 10, 2),
        (2.5000001, 10, 3),
        # Clamping to valid range.
        (-5.0, 10, 0),
        (100.0, 10, 9),
        (9.4, 10, 9),
    ],
)
def test_round_world_to_voxel(raw, size, expected):
    assert round_world_to_voxel(raw, size) == expected


def test_round_world_to_voxel_returns_python_int():
    result = round_world_to_voxel(np.float64(3.5), 10)
    assert isinstance(result, int)


# ── cross-path equality at level 0 ────────────────────────────────────────
#
# In-memory uses ``_transform_slice_indices`` with a data->world transform and
# ``imap_coordinates`` (world->data).  Multiscale uses
# ``_build_axis_selections_multiscale``
# with a world->level-k transform and ``map_coordinates`` (forward).  At level 0
# the multiscale world->level transform is exactly the inverse of the in-memory
# data->world transform, so both must produce identical sliced voxel indices.


def _multiscale_index_for_axis(
    world_to_level0: AffineTransform,
    ndim: int,
    sliced_axis: int,
    slice_indices: dict[int, int],
    level_shape: tuple[int, ...],
    displayed_axes: tuple[int, ...],
) -> int:
    """Return the multiscale-path voxel index for a single sliced axis."""
    sel = AxisAlignedSelectionState(
        displayed_axes=displayed_axes,
        slice_indices=slice_indices,
    )
    # Display ranges are irrelevant to the sliced axis; full extent is fine.
    display_coords = [(0, level_shape[ax]) for ax in displayed_axes]
    axis_selections = _build_axis_selections_multiscale(
        sel,
        ndim,
        display_coords,
        level_shape=level_shape,
        world_to_level_k=world_to_level0,
    )
    value = axis_selections[sliced_axis]
    assert isinstance(value, int), "sliced axis must collapse to a scalar"
    return value


@pytest.mark.parametrize(
    "data_to_world",
    [
        AffineTransform.identity(ndim=3),
        AffineTransform.from_scale((4.0, 1.0, 1.0)),  # anisotropic z
        AffineTransform.from_scale((2.0, 2.0, 2.0)),  # isotropic 2x
        AffineTransform.from_scale_and_translation(
            scale=(3.0, 1.0, 1.0), translation=(0.5, 0.0, 0.0)
        ),
    ],
)
@pytest.mark.parametrize("world_pos", [0, 1, 5, 7, 10, 11])
def test_in_memory_and_multiscale_agree_3d(data_to_world, world_pos):
    """Axis 0 sliced, axes (1, 2) displayed; both paths pick the same voxel."""
    store_shape = (12, 8, 8)
    ndim = 3
    sliced_axis = 0
    displayed_axes = (1, 2)
    slice_indices = {sliced_axis: world_pos}

    in_memory = _transform_slice_indices(slice_indices, data_to_world, store_shape)[
        sliced_axis
    ]

    world_to_level0 = AffineTransform(matrix=data_to_world.inverse_matrix)
    multiscale = _multiscale_index_for_axis(
        world_to_level0,
        ndim,
        sliced_axis,
        slice_indices,
        store_shape,
        displayed_axes,
    )

    assert in_memory == multiscale


@pytest.mark.parametrize("world_pos", [3, 4, 5])
def test_half_integer_tie_rounds_up_both_paths(world_pos):
    """A 2x scale maps even world positions to half-integer voxel coords.

    world=3 -> data 1.5 -> 2; world=4 -> data 2.0 -> 2; world=5 -> data 2.5 -> 3.
    Both paths must agree and round half up.
    """
    store_shape = (12, 8, 8)
    data_to_world = AffineTransform.from_scale((2.0, 1.0, 1.0))
    expected = round_world_to_voxel(world_pos / 2.0, store_shape[0])

    in_memory = _transform_slice_indices({0: world_pos}, data_to_world, store_shape)[0]

    world_to_level0 = AffineTransform(matrix=data_to_world.inverse_matrix)
    multiscale = _multiscale_index_for_axis(
        world_to_level0, 3, 0, {0: world_pos}, store_shape, (1, 2)
    )

    assert in_memory == multiscale == expected


def test_multiple_sliced_axes_agree_4d():
    """ndim=4 with axes 0 and 1 sliced, axes (2, 3) displayed."""
    store_shape = (10, 6, 8, 8)
    ndim = 4
    displayed_axes = (2, 3)
    data_to_world = AffineTransform.from_scale((2.0, 3.0, 1.0, 1.0))
    slice_indices = {0: 7, 1: 5}

    in_memory = _transform_slice_indices(slice_indices, data_to_world, store_shape)

    world_to_level0 = AffineTransform(matrix=data_to_world.inverse_matrix)
    for sliced_axis in (0, 1):
        multiscale = _multiscale_index_for_axis(
            world_to_level0,
            ndim,
            sliced_axis,
            slice_indices,
            store_shape,
            displayed_axes,
        )
        assert in_memory[sliced_axis] == multiscale
