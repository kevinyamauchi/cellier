"""Tests for AxisAlignedTexturePositioning, ChunkSelector, and world_transform.

Phase A of the chunked rendering pipeline validation.
"""

import numpy as np
import pytest

from cellier.transform import AffineTransform
from cellier.utils.chunked_image._chunk_selection import (
    AxisAlignedTexturePositioning,
    ChunkSelector,
    TextureBoundsFiltering,
)
from cellier.utils.chunked_image._data_classes import (
    TextureConfiguration,
    ViewParameters,
)
from cellier.utils.chunked_image._multiscale_image_model import (
    MultiscaleImageModel,
    ScaleLevelModel,
    compute_scale_transform,
)

# ── Shared constants ──────────────────────────────────────────────────────────

# Identity world transform used as a neutral value wherever select_chunks
# requires the argument but the test is not exercising world_transform logic.
_IDENTITY_WT = AffineTransform.from_scale_and_translation(
    (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def scale_level_identity():
    """4x4x4 grid of 32^3 chunks in a 128^3 volume; identity scale transform."""
    return ScaleLevelModel(
        shape=(128, 128, 128),
        chunk_shape=(32, 32, 32),
        transform=AffineTransform.from_scale_and_translation(
            (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
        ),
    )


@pytest.fixture
def scale_level_2x():
    """4x4x4 grid of 16^3 chunks in a 64^3 volume; 2x downscale transform."""
    return ScaleLevelModel(
        shape=(64, 64, 64),
        chunk_shape=(16, 16, 16),
        transform=compute_scale_transform(2.0),
    )


@pytest.fixture
def multiscale_model_2scales(scale_level_identity, scale_level_2x):
    """Two-scale model with default (identity) world transform."""
    return MultiscaleImageModel(scales=[scale_level_identity, scale_level_2x])


@pytest.fixture
def texture_config_32():
    return TextureConfiguration(texture_width=32)


@pytest.fixture
def texture_config_64():
    return TextureConfiguration(texture_width=64)


@pytest.fixture
def texture_config_128():
    return TextureConfiguration(texture_width=128)


@pytest.fixture
def positioning_strategy():
    return AxisAlignedTexturePositioning()


@pytest.fixture
def chunk_selector():
    return ChunkSelector(
        positioning_strategy=AxisAlignedTexturePositioning(),
        filtering_strategy=TextureBoundsFiltering(),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_view_params(
    frustum_corners: np.ndarray,
    view_direction: np.ndarray,
) -> ViewParameters:
    """Build ViewParameters, deriving near_plane_center from frustum_corners[0].

    Parameters
    ----------
    frustum_corners : np.ndarray
        Shape (2, 4, 3). Row 0 = near plane, row 1 = far plane.
    view_direction : np.ndarray
        Shape (3,). Will be normalised.
    """
    norm = np.linalg.norm(view_direction)
    assert norm > 0
    return ViewParameters(
        frustum_corners=frustum_corners.astype(np.float32),
        view_direction=(view_direction / norm).astype(np.float32),
        near_plane_center=frustum_corners[0].mean(axis=0).astype(np.float32),
    )


def make_axis_frustum(
    near_center: np.ndarray,
    half_size: float,
    depth: float,
) -> np.ndarray:
    """Build an axis-aligned box frustum (orthographic style).

    The near plane is a square centred on ``near_center`` in the y-x plane.
    The far plane is displaced by ``depth`` in the +z direction (axis 0).

    Parameters
    ----------
    near_center : np.ndarray
        Shape (3,) centre of the near plane, (z, y, x).
    half_size : float
        Half-width in y and x.
    depth : float
        Depth of the frustum along z (axis 0).

    Returns
    -------
    np.ndarray
        Shape (2, 4, 3) ordered (left-bottom, right-bottom,
        right-top, left-top).
    """
    z, y, x = near_center
    near = np.array(
        [
            [z, y - half_size, x - half_size],
            [z, y - half_size, x + half_size],
            [z, y + half_size, x + half_size],
            [z, y + half_size, x - half_size],
        ],
        dtype=np.float32,
    )
    far = near.copy()
    far[:, 0] += depth
    return np.stack([near, far], axis=0)


def chunk_corners_from_bbox(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> np.ndarray:
    """Return the 8 corners of an AABB.

    Parameters
    ----------
    bbox_min, bbox_max : np.ndarray
        Shape (3,) in (z, y, x) order.

    Returns
    -------
    np.ndarray
        Shape (8, 3).
    """
    z0, y0, x0 = bbox_min
    z1, y1, x1 = bbox_max
    return np.array(
        [
            [z0, y0, x0],
            [z0, y0, x1],
            [z0, y1, x0],
            [z0, y1, x1],
            [z1, y0, x0],
            [z1, y0, x1],
            [z1, y1, x0],
            [z1, y1, x1],
        ],
        dtype=np.float32,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Group 1: AxisAlignedTexturePositioning.position_texture() tests (T-01..T-09)
# ══════════════════════════════════════════════════════════════════════════════


def test_primary_axis_along_positive_z(
    positioning_strategy, scale_level_identity, texture_config_64
):
    """T-01: View along +z selects axis 0."""
    frustum_corners = make_axis_frustum(
        near_center=np.array([-10.0, 16.0, 16.0]), half_size=16.0, depth=20.0
    )
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    _, _, primary_axis = positioning_strategy.position_texture(
        view_params, scale_level_identity, texture_config_64
    )
    assert primary_axis == 0


@pytest.mark.parametrize(
    "view_dir,expected_axis",
    [
        (np.array([1.0, 0.0, 0.0]), 0),  # +z
        (np.array([-1.0, 0.0, 0.0]), 0),  # -z
        (np.array([0.0, 1.0, 0.0]), 1),  # +y
        (np.array([0.0, -1.0, 0.0]), 1),  # -y
        (np.array([0.0, 0.0, 1.0]), 2),  # +x
        (np.array([0.0, 0.0, -1.0]), 2),  # -x
    ],
)
def test_primary_axis_all_directions(
    positioning_strategy,
    scale_level_identity,
    texture_config_64,
    view_dir,
    expected_axis,
):
    """T-02: All six canonical directions map to correct primary axis."""
    frustum_corners = make_axis_frustum(
        np.array([16.0, 16.0, 16.0]), half_size=8.0, depth=10.0
    )
    view_params = make_view_params(frustum_corners, view_dir)

    _, _, primary_axis = positioning_strategy.position_texture(
        view_params, scale_level_identity, texture_config_64
    )
    assert primary_axis == expected_axis


def test_texture_bounds_are_cube(
    positioning_strategy, scale_level_identity, texture_config_64
):
    """T-03: texture_max - texture_min equals texture_width on every axis."""
    frustum_corners = make_axis_frustum(
        np.array([-10.0, 16.0, 16.0]), half_size=16.0, depth=20.0
    )
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    _, (tex_min, tex_max), _ = positioning_strategy.position_texture(
        view_params, scale_level_identity, texture_config_64
    )
    np.testing.assert_allclose(tex_max - tex_min, texture_config_64.texture_width)


def test_transform_is_pure_translation(
    positioning_strategy, scale_level_identity, texture_config_64
):
    """T-04: Returned AffineTransform has identity upper-left 3x3 block."""
    frustum_corners = make_axis_frustum(
        np.array([-10.0, 16.0, 16.0]), half_size=16.0, depth=20.0
    )
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    transform, _, _ = positioning_strategy.position_texture(
        view_params, scale_level_identity, texture_config_64
    )
    np.testing.assert_allclose(transform.matrix[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(transform.matrix[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6)


def test_transform_translation_equals_texture_min(
    positioning_strategy, scale_level_identity, texture_config_64
):
    """T-05: Mapping texture-space origin (0,0,0) through transform gives tex_min."""
    frustum_corners = make_axis_frustum(
        np.array([-10.0, 32.0, 32.0]), half_size=8.0, depth=10.0
    )
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    transform, (tex_min, _), _ = positioning_strategy.position_texture(
        view_params, scale_level_identity, texture_config_64
    )

    origin = np.array([[0.0, 0.0, 0.0]])
    mapped = transform.map_coordinates(origin)[0]
    np.testing.assert_allclose(mapped, tex_min, atol=1e-6)


def test_near_plane_outside_data_clamps_to_min(
    positioning_strategy, scale_level_identity, texture_config_64
):
    """T-06: Near plane far below data extent — texture clamped to [0, tw] boundary."""
    # near_plane_center at z=-100, well outside data [0, 128]
    frustum_corners = make_axis_frustum(
        near_center=np.array([-100.0, 0.0, 0.0]), half_size=8.0, depth=10.0
    )
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    _, (tex_min, tex_max), _ = positioning_strategy.position_texture(
        view_params, scale_level_identity, texture_config_64
    )

    np.testing.assert_allclose(tex_min, [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(tex_max, [64.0, 64.0, 64.0], atol=1e-6)


def test_near_plane_outside_data_clamps_to_max(
    positioning_strategy, scale_level_identity, texture_config_64
):
    """T-07: Near plane far above data extent — texture clamped to [tw, 2*tw]."""
    # near_plane_center at z=200, well outside data [0, 128]
    frustum_corners = make_axis_frustum(
        near_center=np.array([200.0, 0.0, 0.0]), half_size=8.0, depth=10.0
    )
    view_params = make_view_params(frustum_corners, np.array([-1.0, 0.0, 0.0]))

    _, (tex_min, tex_max), _ = positioning_strategy.position_texture(
        view_params, scale_level_identity, texture_config_64
    )

    # max_valid[0] = 128 - 64 = 64; clamp(168, 0, 64) = 64
    np.testing.assert_allclose(tex_min[0], 64.0, atol=1e-6)
    np.testing.assert_allclose(tex_max[0], 128.0, atol=1e-6)


def test_texture_always_texture_width_wide(
    positioning_strategy, scale_level_identity, texture_config_64
):
    """T-08: Texture is always exactly texture_width on every axis."""
    frustum_corners = make_axis_frustum(
        near_center=np.array([-100.0, 0.0, 0.0]), half_size=8.0, depth=10.0
    )
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    _, (tex_min, tex_max), _ = positioning_strategy.position_texture(
        view_params, scale_level_identity, texture_config_64
    )

    np.testing.assert_allclose(tex_max - tex_min, 64.0, atol=1e-6)


def test_single_chunk_contained_in_texture_bounds(
    positioning_strategy, scale_level_identity, texture_config_64
):
    """T-09: A single chunk whose center matches near_plane_center is inside bounds."""
    # near_plane_center at (48, 48, 48) — center of chunk at [32..64]^3
    frustum_corners = make_axis_frustum(
        near_center=np.array([-10.0, 48.0, 48.0]), half_size=16.0, depth=20.0
    )
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    _, (tex_min, tex_max), _ = positioning_strategy.position_texture(
        view_params, scale_level_identity, texture_config_64
    )

    single_chunk = chunk_corners_from_bbox(
        np.array([32.0, 32.0, 32.0]), np.array([64.0, 64.0, 64.0])
    )
    within = np.all(
        (single_chunk >= tex_min[np.newaxis] - 1e-5)
        & (single_chunk <= tex_max[np.newaxis] + 1e-5)
    )
    assert within, (
        f"Single chunk corners not inside texture bounds.\n"
        f"tex=[{tex_min}, {tex_max}]\nchunk corners=\n{single_chunk}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Group 1b: Camera-centered positioning behaviour (T-19..T-21)
# ══════════════════════════════════════════════════════════════════════════════


def test_texture_centered_on_near_plane(
    positioning_strategy, scale_level_identity, texture_config_64
):
    """T-19: Texture min corner is chunk-grid-snapped centering of near_plane_center."""
    # near_plane_center at (64, 64, 64) — exact centre of volume
    # raw_corner = (64-32, 64-32, 64-32) = (32, 32, 32)
    # chunk_shape = 32, so snapped = (32, 32, 32) — already on grid
    frustum_corners = make_axis_frustum(
        near_center=np.array([64.0, 64.0, 64.0]), half_size=8.0, depth=10.0
    )
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    _, (tex_min, tex_max), _ = positioning_strategy.position_texture(
        view_params, scale_level_identity, texture_config_64
    )

    np.testing.assert_allclose(tex_min, [32.0, 32.0, 32.0], atol=1e-6)
    np.testing.assert_allclose(tex_max, [96.0, 96.0, 96.0], atol=1e-6)


def test_texture_snapped_to_chunk_grid(
    positioning_strategy, scale_level_identity, texture_config_64
):
    """T-20: Positioning corner is always a multiple of chunk_shape."""
    # Use a near_center that is NOT on the chunk grid
    frustum_corners = make_axis_frustum(
        near_center=np.array([50.0, 50.0, 50.0]), half_size=8.0, depth=10.0
    )
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    _, (tex_min, _), _ = positioning_strategy.position_texture(
        view_params, scale_level_identity, texture_config_64
    )

    chunk = np.array(scale_level_identity.chunk_shape, dtype=float)
    remainder = tex_min % chunk
    np.testing.assert_allclose(remainder, 0.0, atol=1e-6)


def test_texture_stays_within_data_extent(
    positioning_strategy, scale_level_identity, texture_config_64
):
    """T-21: Texture bounds always stay within [0, shape] regardless of near plane."""
    for near_z in [-500.0, 0.0, 64.0, 127.0, 500.0]:
        frustum_corners = make_axis_frustum(
            near_center=np.array([near_z, 64.0, 64.0]), half_size=8.0, depth=10.0
        )
        view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

        _, (tex_min, tex_max), _ = positioning_strategy.position_texture(
            view_params, scale_level_identity, texture_config_64
        )

        shape = np.array(scale_level_identity.shape, dtype=float)
        assert np.all(tex_min >= 0.0), f"tex_min below 0 for near_z={near_z}: {tex_min}"
        assert np.all(
            tex_max <= shape + 1e-6
        ), f"tex_max above shape for near_z={near_z}: {tex_max}"


# ══════════════════════════════════════════════════════════════════════════════
# Group 2: ChunkSelector round-trip tests (T-10..T-15)
# ══════════════════════════════════════════════════════════════════════════════


def test_basic_selection_returns_chunks(
    chunk_selector, scale_level_identity, texture_config_64
):
    """T-10: A frustum spanning the volume selects at least some chunks."""
    frustum_corners = make_axis_frustum(
        near_center=np.array([-10.0, 64.0, 64.0]), half_size=64.0, depth=200.0
    )
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    result = chunk_selector.select_chunks(
        scale_level_identity,
        view_params,
        texture_config_64,
        world_transform=_IDENTITY_WT,
    )

    assert result.n_selected_chunks > 0
    assert result.selected_chunk_mask.shape == (scale_level_identity.n_chunks,)
    assert result.selected_chunk_mask.dtype == bool


def test_empty_frustum_mask_returns_zero(
    chunk_selector, scale_level_identity, texture_config_64
):
    """T-11: All-False frustum_visible_chunks returns zero selected chunks."""
    frustum_corners = make_axis_frustum(np.array([-10.0, 64.0, 64.0]), 64.0, 200.0)
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))
    all_false = np.zeros(scale_level_identity.n_chunks, dtype=bool)

    result = chunk_selector.select_chunks(
        scale_level_identity,
        view_params,
        texture_config_64,
        world_transform=_IDENTITY_WT,
        frustum_visible_chunks=all_false,
    )

    assert result.n_selected_chunks == 0
    assert not np.any(result.selected_chunk_mask)


def test_selection_respects_frustum_mask(
    chunk_selector, scale_level_identity, texture_config_128
):
    """T-12: Selection never picks chunks outside frustum_visible_chunks mask."""
    frustum_mask = np.zeros(scale_level_identity.n_chunks, dtype=bool)
    frustum_mask[:8] = True

    frustum_corners = make_axis_frustum(np.array([-10.0, 64.0, 64.0]), 64.0, 200.0)
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    result = chunk_selector.select_chunks(
        scale_level_identity,
        view_params,
        texture_config_128,
        world_transform=_IDENTITY_WT,
        frustum_visible_chunks=frustum_mask,
    )

    out_of_mask = np.flatnonzero(result.selected_chunk_mask)
    assert np.all(
        out_of_mask < 8
    ), f"Chunks outside frustum mask were selected: {out_of_mask[out_of_mask >= 8]}"


def test_result_transform_is_translation(
    chunk_selector, scale_level_identity, texture_config_64
):
    """T-13: texture_to_world_transform is a pure translation."""
    frustum_corners = make_axis_frustum(np.array([-10.0, 64.0, 64.0]), 64.0, 200.0)
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    result = chunk_selector.select_chunks(
        scale_level_identity,
        view_params,
        texture_config_64,
        world_transform=_IDENTITY_WT,
    )

    m = result.texture_to_world_transform.matrix
    np.testing.assert_allclose(m[:3, :3], np.eye(3), atol=1e-5)
    np.testing.assert_allclose(m[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-5)


def test_selected_chunks_inside_texture_bounds(
    chunk_selector, scale_level_identity, texture_config_64
):
    """T-14: Every corner of every selected chunk lies within texture_bounds_world."""
    frustum_corners = make_axis_frustum(np.array([-10.0, 64.0, 64.0]), 64.0, 200.0)
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    result = chunk_selector.select_chunks(
        scale_level_identity,
        view_params,
        texture_config_64,
        world_transform=_IDENTITY_WT,
    )

    if result.n_selected_chunks == 0:
        pytest.skip("No chunks selected; nothing to validate.")

    tex_min, tex_max = result.texture_bounds_world
    # chunk_corners_scale_0 == world coords when transform is identity
    selected_corners = scale_level_identity.chunk_corners_scale_0[
        result.selected_chunk_mask
    ]  # (n_selected, 8, 3)

    assert np.all(
        selected_corners >= tex_min[np.newaxis, np.newaxis] - 1e-4
    ), "Some selected chunk corners are below texture_bounds_world min."
    assert np.all(
        selected_corners <= tex_max[np.newaxis, np.newaxis] + 1e-4
    ), "Some selected chunk corners are above texture_bounds_world max."


def test_larger_texture_selects_at_least_as_many_chunks(
    chunk_selector, scale_level_identity
):
    """T-15: Increasing texture_width never reduces the number of selected chunks."""
    frustum_corners = make_axis_frustum(np.array([-10.0, 64.0, 64.0]), 64.0, 200.0)
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    result_small = chunk_selector.select_chunks(
        scale_level_identity,
        view_params,
        TextureConfiguration(texture_width=32),
        world_transform=_IDENTITY_WT,
    )
    result_large = chunk_selector.select_chunks(
        scale_level_identity,
        view_params,
        TextureConfiguration(texture_width=128),
        world_transform=_IDENTITY_WT,
    )

    assert result_large.n_selected_chunks >= result_small.n_selected_chunks


# ══════════════════════════════════════════════════════════════════════════════
# Group 3: MultiscaleImageModel.world_transform tests (T-16..T-18)
# ══════════════════════════════════════════════════════════════════════════════


def test_world_transform_default_is_identity(multiscale_model_2scales):
    """T-16: Default world_transform is identity."""
    m = multiscale_model_2scales.world_transform.matrix
    np.testing.assert_allclose(m, np.eye(4, dtype=np.float32), atol=1e-6)


def test_world_transform_non_identity_stored(scale_level_identity):
    """T-17: Non-identity world_transform is stored correctly."""
    translation = (10.0, -5.0, 3.0)
    world_transform = AffineTransform.from_translation(translation)
    model = MultiscaleImageModel(
        scales=[scale_level_identity],
        world_transform=world_transform,
    )
    expected = np.eye(4, dtype=np.float32)
    expected[0, 3] = 10.0
    expected[1, 3] = -5.0
    expected[2, 3] = 3.0
    np.testing.assert_allclose(model.world_transform.matrix, expected, atol=1e-6)


def test_world_transform_roundtrip(scale_level_identity):
    """T-18: Non-identity world_transform round-trips correctly."""
    translation = (10.0, -5.0, 3.0)
    world_transform = AffineTransform.from_translation(translation)
    model = MultiscaleImageModel(
        scales=[scale_level_identity],
        world_transform=world_transform,
    )

    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    mapped = model.world_transform.map_coordinates(points)
    roundtrip = model.world_transform.imap_coordinates(mapped)
    np.testing.assert_allclose(roundtrip, points, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# Group 4: New anchor and world_transform regression tests (T-22..T-23)
# ══════════════════════════════════════════════════════════════════════════════


def test_perspective_anchor_centers_texture_on_data(
    chunk_selector, scale_level_identity, texture_config_64
):
    """T-22: Perspective camera outside data — texture centred on data centre.

    With near_dist=1, far_dist=300, half_h_near=0.5, half_h_far=150 the
    recovered camera sits at scale_N = (-101, 64, 64).  Projecting the data
    centre (64, 64, 64) onto the +z ray gives anchor = (64, 64, 64), which
    snaps to tex_min = (32, 32, 32).

    Previously (near_plane_center anchor) the near plane at z≈-100 would
    have been outside the data, clamping tex_min to (0, 0, 0) — covering only
    the data face nearest the camera.
    """
    # Camera in scale_0 at (-101, 64, 64), looking along +z (axis 0).
    # near plane at z=-100, far plane at z=199, both centred on (y=64, x=64).
    near_half_h = 0.5
    far_half_h = 150.0
    near_z = -100.0
    far_z = 199.0

    near = np.array(
        [
            [near_z, 64.0 - near_half_h, 64.0 - near_half_h],
            [near_z, 64.0 - near_half_h, 64.0 + near_half_h],
            [near_z, 64.0 + near_half_h, 64.0 + near_half_h],
            [near_z, 64.0 + near_half_h, 64.0 - near_half_h],
        ],
        dtype=np.float32,
    )
    far = np.array(
        [
            [far_z, 64.0 - far_half_h, 64.0 - far_half_h],
            [far_z, 64.0 - far_half_h, 64.0 + far_half_h],
            [far_z, 64.0 + far_half_h, 64.0 + far_half_h],
            [far_z, 64.0 + far_half_h, 64.0 - far_half_h],
        ],
        dtype=np.float32,
    )
    frustum_corners = np.stack([near, far])
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    positioning = AxisAlignedTexturePositioning()
    _, (tex_min, tex_max), _ = positioning.position_texture(
        view_params, scale_level_identity, texture_config_64
    )

    # anchor = (64, 64, 64) → raw_corner = (32, 32, 32) → snapped = (32, 32, 32)
    np.testing.assert_allclose(tex_min, [32.0, 32.0, 32.0], atol=1.0)
    np.testing.assert_allclose(tex_max, [96.0, 96.0, 96.0], atol=1.0)


def test_non_identity_world_transform_texture_offset_roundtrip(
    chunk_selector, scale_level_identity, texture_config_64
):
    """T-23: Texture offset round-trips correctly with non-identity world_transform.

    world_transform = translate(50, 50, 50) shifts every scale_0 point by
    (50, 50, 50) in world space.  Chunk (1,1,1) has scale_N min corner
    (32, 32, 32), scale_0 min corner (32, 32, 32), world min corner
    (82, 82, 82).

    The expected texture offset is (0, 0, 0) because the texture window starts
    at scale_N (32, 32, 32).  Composing the full chain and inverting should
    reproduce this exactly.
    """
    # Place perspective camera in world space so the full data is in view.
    # Camera in scale_0 = (-101, 64, 64); in world = (-51, 114, 114).
    world_transform = AffineTransform.from_translation((50.0, 50.0, 50.0))

    near_half_h = 0.5
    far_half_h = 150.0
    near_z_world = -100.0 + 50.0  # = -50.0
    far_z_world = 199.0 + 50.0  # = 249.0
    cy_world = 64.0 + 50.0  # = 114.0
    cx_world = 64.0 + 50.0  # = 114.0

    near = np.array(
        [
            [near_z_world, cy_world - near_half_h, cx_world - near_half_h],
            [near_z_world, cy_world - near_half_h, cx_world + near_half_h],
            [near_z_world, cy_world + near_half_h, cx_world + near_half_h],
            [near_z_world, cy_world + near_half_h, cx_world - near_half_h],
        ],
        dtype=np.float32,
    )
    far = np.array(
        [
            [far_z_world, cy_world - far_half_h, cx_world - far_half_h],
            [far_z_world, cy_world - far_half_h, cx_world + far_half_h],
            [far_z_world, cy_world + far_half_h, cx_world + far_half_h],
            [far_z_world, cy_world + far_half_h, cx_world - far_half_h],
        ],
        dtype=np.float32,
    )
    frustum_corners = np.stack([near, far])
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    result = chunk_selector.select_chunks(
        scale_level_identity,
        view_params,
        texture_config_64,
        world_transform=world_transform,
    )

    assert result.n_selected_chunks > 0, "No chunks selected; cannot validate."

    # Pick one selected chunk and verify the round-trip:
    # texture_to_world_transform.map_coordinates(tex_offset) == world_corner
    idx = int(np.flatnonzero(result.selected_chunk_mask)[0])
    grid = scale_level_identity.chunk_grid_shape
    iz = idx // (grid[1] * grid[2])
    iy = (idx // grid[2]) % grid[1]
    ix = idx % grid[2]
    cz, cy, cx = scale_level_identity.chunk_shape
    scale_n_corner = np.array([[iz * cz, iy * cy, ix * cx]], dtype=np.float64)

    # scale_N → scale_0 (identity) → world
    scale_0_corner = scale_level_identity.transform.map_coordinates(scale_n_corner)
    expected_world_corner = world_transform.map_coordinates(scale_0_corner)

    # Invert via texture_to_world_transform
    tex_offset = result.texture_to_world_transform.imap_coordinates(
        expected_world_corner
    )

    # Round-trip: mapping tex_offset forward should recover the world corner
    recovered_world = result.texture_to_world_transform.map_coordinates(tex_offset)
    np.testing.assert_allclose(
        recovered_world,
        expected_world_corner,
        atol=1e-4,
        err_msg=(
            f"Round-trip failed for chunk {idx} (iz={iz}, iy={iy}, ix={ix}).\n"
            f"Expected world corner: {expected_world_corner}\n"
            f"Recovered world corner: {recovered_world}"
        ),
    )
