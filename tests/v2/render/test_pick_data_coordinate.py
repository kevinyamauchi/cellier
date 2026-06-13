"""Tests for the shared pick → data-coordinate helpers.

These lock in the invariant the whole picking pipeline relies on: the decoded
coordinate is in level-0 data space and ``floor`` of each component recovers the
integer voxel index — uniformly for memory and multiscale, image and labels,
2-D and 3-D.
"""

from __future__ import annotations

import math

import pytest

from cellier.render.visuals._pick import (
    memory_image_data_coordinate,
    multiscale_image_data_coordinate,
    multiscale_volume_data_coordinate,
)

# ── Memory image / labels (real data texture) ──────────────────────────────


def test_memory_image_2d_center_floors_to_index() -> None:
    # pygfx reports pixel center at integer index, fractional in [-0.5, 0.5).
    pick = {"index": (3, 7), "pixel_coord": (0.0, 0.0)}
    coord = memory_image_data_coordinate(pick, ndim=2)
    assert coord == (3.5, 7.5)
    assert tuple(math.floor(c) for c in coord) == (3, 7)


def test_memory_image_2d_subpixel_stays_in_voxel() -> None:
    # Anywhere within the pixel (frac in [-0.5, 0.5)) must floor to the index.
    for fx in (-0.5, -0.1, 0.0, 0.49):
        pick = {"index": (3, 7), "pixel_coord": (fx, fx)}
        coord = memory_image_data_coordinate(pick, ndim=2)
        assert tuple(math.floor(c) for c in coord) == (3, 7)


def test_memory_volume_3d_center_floors_to_index() -> None:
    pick = {"index": (2, 5, 9), "voxel_coord": (0.0, 0.0, 0.0)}
    coord = memory_image_data_coordinate(pick, ndim=3)
    assert coord == (2.5, 5.5, 9.5)
    assert tuple(math.floor(c) for c in coord) == (2, 5, 9)


def test_memory_image_missing_index_returns_none() -> None:
    assert memory_image_data_coordinate({}, ndim=2) is None


def test_memory_image_missing_frac_defaults_to_center() -> None:
    coord = memory_image_data_coordinate({"index": (1, 1)}, ndim=2)
    assert coord == (1.5, 1.5)


# ── Multiscale volume (NormSizedVolume norm_pos) ───────────────────────────


def test_multiscale_volume_center_floors_to_index() -> None:
    # dataset 4x8x16 (x, y, z); norm_size cancels out. A norm_pos at the centre
    # of voxel i corresponds to n = (i + 0.5)/size, i.e. norm_pos/norm_size +
    # 0.5 = (i + 0.5)/size.
    norm_size = (2.0, 4.0, 8.0)
    dataset_size = (4.0, 8.0, 16.0)
    # target voxel (1, 3, 10): n = (i+0.5)/size -> norm_pos = (n - 0.5)*norm_size
    target = (1, 3, 10)
    norm_pos = tuple(
        ((t + 0.5) / s - 0.5) * ns for t, s, ns in zip(target, dataset_size, norm_size)
    )
    coord = multiscale_volume_data_coordinate(
        {"norm_pos": norm_pos}, norm_size, dataset_size
    )
    assert tuple(math.floor(c) for c in coord) == target


def test_multiscale_volume_edge_clamps_in_range() -> None:
    # n == 1 (upper face) would map to exactly dataset_size; must clamp so floor
    # stays at size - 1 rather than going out of range.
    norm_size = (2.0, 2.0, 2.0)
    dataset_size = (4.0, 4.0, 4.0)
    norm_pos = tuple(0.5 * ns for ns in norm_size)  # n = 1.0
    coord = multiscale_volume_data_coordinate(
        {"norm_pos": norm_pos}, norm_size, dataset_size
    )
    assert tuple(math.floor(c) for c in coord) == (3, 3, 3)


def test_multiscale_volume_missing_norm_pos_returns_none() -> None:
    assert multiscale_volume_data_coordinate({}, (1, 1, 1), (1, 1, 1)) is None


# ── Multiscale 2-D image / labels (tile proxy) ─────────────────────────────


def test_multiscale_image_resolves_within_tile_not_to_tile_center() -> None:
    # Level-0 shape 1000x2000 (H, W) split into 4x8 tiles of ~256 px. A pick in
    # proxy tile (gx=5, gy=2) at sub-tile offset must land on a real pixel, not
    # the tile centre — this is the bug the change fixes.
    level0 = (1000, 2000)  # (H, W)
    grid = (4, 8)  # (gh, gw)
    pick = {"index": (5, 2), "pixel_coord": (0.25, -0.25)}
    coord = multiscale_image_data_coordinate(pick, level0, grid)
    # data-x per proxy texel = W/gw = 250; data-y per texel = H/gh = 250.
    # x = (5 + 0.25 + 0.5) * 250 = 1437.5 ; y = (2 - 0.25 + 0.5) * 250 = 562.5
    assert coord == pytest.approx((1437.5, 562.5))
    assert tuple(math.floor(c) for c in coord) == (1437, 562)


def test_multiscale_image_first_tile_origin() -> None:
    level0 = (512, 512)
    grid = (2, 2)  # 256 px per texel
    pick = {"index": (0, 0), "pixel_coord": (-0.5, -0.5)}
    coord = multiscale_image_data_coordinate(pick, level0, grid)
    # (0 - 0.5 + 0.5) * 256 = 0 on both axes.
    assert coord == pytest.approx((0.0, 0.0))
    assert tuple(math.floor(c) for c in coord) == (0, 0)


def test_multiscale_image_clamps_upper_edge() -> None:
    level0 = (256, 256)
    grid = (1, 1)  # one tile, 256 px per texel
    # frac just under 0.5 -> (0 + 0.4999 + 0.5) * 256 ≈ 255.97 -> floor 255.
    pick = {"index": (0, 0), "pixel_coord": (0.4999, 0.4999)}
    coord = multiscale_image_data_coordinate(pick, level0, grid)
    assert all(0.0 <= c < 256.0 for c in coord)
    assert tuple(math.floor(c) for c in coord) == (255, 255)


def test_multiscale_image_missing_index_returns_none() -> None:
    assert multiscale_image_data_coordinate({}, (10, 10), (1, 1)) is None
