"""Tests for the MultiscaleVolumeBrick shader pipeline.

Tests A-D from the implementation plan: coordinate round-trips,
world transform composition, and block-scale convention correctness.
"""

from __future__ import annotations

import numpy as np

from cellier.v2.render.shaders._multiscale_volume_brick import (
    compose_world_transform,
    compute_normalized_size,
)

# ── Coordinate helpers (Python mirrors of WGSL functions) ──────────────


def norm_to_voxel(pos, norm_size, dataset_size):
    """Normalized space -> finest-level voxel coordinates."""
    return ((pos / norm_size) + 0.5) * dataset_size


def voxel_to_norm(voxel, norm_size, dataset_size):
    """Finest-level voxel -> normalized space."""
    return (voxel / dataset_size - 0.5) * norm_size


# ── Test A: world transform round-trip ─────────────────────────────────


def test_world_transform_round_trip():
    """Verify that normalized corners map to the correct data coordinates."""
    dataset_size = np.array([512.0, 256.0, 128.0])
    spacing = np.array([1.0, 1.0, 1.0])
    phys_size = dataset_size * spacing
    norm_size = phys_size / phys_size.max()

    data_to_world = np.eye(4, dtype=np.float32)  # identity

    M = compose_world_transform(data_to_world, dataset_size, norm_size)

    # Normalized corner [-h, -h, -h] → data origin [0, 0, 0]
    h = norm_size / 2
    corner_norm = np.array([-h[0], -h[1], -h[2], 1.0])
    data_origin = M @ corner_norm
    np.testing.assert_allclose(data_origin[:3], [0.0, 0.0, 0.0], atol=1e-5)

    # Normalized corner [+h, +h, +h] → data max [512, 256, 128]
    corner_norm = np.array([h[0], h[1], h[2], 1.0])
    data_max = M @ corner_norm
    np.testing.assert_allclose(data_max[:3], dataset_size, atol=1e-5)


def test_world_transform_anisotropic_spacing():
    """Verify transform with anisotropic voxel spacing."""
    dataset_size = np.array([400.0, 400.0, 200.0])  # x, y, z
    spacing = np.array([1.0, 1.0, 2.0])  # z-axis has 2x spacing
    norm_size = compute_normalized_size(dataset_size, spacing)

    # Physical extents: [400, 400, 400] — all equal, so norm_size = [1, 1, 1]
    np.testing.assert_allclose(norm_size, [1.0, 1.0, 1.0], atol=1e-6)

    data_to_world = np.eye(4, dtype=np.float32)
    M = compose_world_transform(data_to_world, dataset_size, norm_size)

    # Corners of normalized box should map to data extent
    h = norm_size / 2
    corner = np.array([-h[0], -h[1], -h[2], 1.0])
    np.testing.assert_allclose((M @ corner)[:3], [0.0, 0.0, 0.0], atol=1e-4)

    corner = np.array([h[0], h[1], h[2], 1.0])
    np.testing.assert_allclose((M @ corner)[:3], dataset_size, atol=1e-4)


# ── Test B: norm_to_voxel / voxel_to_norm inverse pair ─────────────────


def test_coordinate_round_trip():
    """Verify that norm↔voxel conversion is a perfect round-trip."""
    norm_size = np.array([1.0, 0.5, 0.25])
    dataset_size = np.array([512.0, 256.0, 128.0])

    rng = np.random.default_rng(0)
    voxels = rng.uniform([0, 0, 0], dataset_size, size=(100, 3))
    for v in voxels:
        np.testing.assert_allclose(
            norm_to_voxel(
                voxel_to_norm(v, norm_size, dataset_size), norm_size, dataset_size
            ),
            v,
            atol=1e-4,
        )


def test_coordinate_boundary_values():
    """Verify corner voxel positions map correctly."""
    norm_size = np.array([1.0, 0.5, 0.25])
    dataset_size = np.array([512.0, 256.0, 128.0])

    # Origin voxel [0, 0, 0] → normalized [-0.5, -0.25, -0.125]
    origin = np.array([0.0, 0.0, 0.0])
    norm = voxel_to_norm(origin, norm_size, dataset_size)
    np.testing.assert_allclose(norm, [-0.5, -0.25, -0.125])

    # Max voxel [512, 256, 128] → normalized [+0.5, +0.25, +0.125]
    np.testing.assert_allclose(
        voxel_to_norm(dataset_size, norm_size, dataset_size),
        [0.5, 0.25, 0.125],
    )


# ── Test C: normalized size computation ────────────────────────────────


def test_compute_normalized_size_isotropic():
    """Isotropic spacing: largest axis gets 1.0, others scale by shape ratio."""
    dataset_size = np.array([512.0, 256.0, 128.0])
    spacing = np.ones(3)
    ns = compute_normalized_size(dataset_size, spacing)
    # Physical sizes: [512, 256, 128]. Max = 512.
    np.testing.assert_allclose(ns, [1.0, 0.5, 0.25])


def test_compute_normalized_size_anisotropic():
    """Anisotropic spacing: physical extent determines normalization."""
    dataset_size = np.array([64.0, 128.0, 256.0])
    spacing = np.array([4.0, 2.0, 1.0])
    ns = compute_normalized_size(dataset_size, spacing)
    # Physical: [256, 256, 256] — all equal → [1, 1, 1]
    np.testing.assert_allclose(ns, [1.0, 1.0, 1.0])


# ── Test D: block-scale convention (pos_in_brick computation) ──────────


def test_block_scale_sample_atlas():
    """At LOD level 1 with isotropic 2x downsampling,
    finest-level voxel [128, 64, 32] should map to level-1 voxel [64, 32, 16].
    """
    lod_scale = np.array([2.0, 2.0, 2.0])
    block_size = np.array([32.0, 32.0, 32.0])
    voxel_pos = np.array([128.0, 64.0, 32.0])
    voxel_k = voxel_pos / lod_scale  # [64, 32, 16]
    pos_in_brick = voxel_k - np.floor(voxel_k / block_size) * block_size
    np.testing.assert_allclose(pos_in_brick, [0.0, 0.0, 16.0])


def test_block_scale_anisotropic_downsampling():
    """Anisotropic downsampling: z not downsampled, x/y downsampled 2x."""
    lod_scale = np.array([1.0, 2.0, 2.0])  # x, y halved; z unchanged
    block_size = np.array([32.0, 32.0, 32.0])
    voxel_pos = np.array([64.0, 64.0, 32.0])

    voxel_k = voxel_pos / lod_scale  # [64, 32, 16]
    pos_in_brick = voxel_k - np.floor(voxel_k / block_size) * block_size
    np.testing.assert_allclose(pos_in_brick, [0.0, 0.0, 16.0])

    # Z axis with scale=1: voxel_k.z = 32 → pos_in_brick.z = 0
    voxel_pos2 = np.array([64.0, 64.0, 64.0])
    voxel_k2 = voxel_pos2 / lod_scale  # [64, 32, 32]
    pos_in_brick2 = voxel_k2 - np.floor(voxel_k2 / block_size) * block_size
    np.testing.assert_allclose(pos_in_brick2, [0.0, 0.0, 0.0])
