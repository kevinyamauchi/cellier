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
    frustum_chunk_corners = chunk_corners_from_bbox(
        np.array([0.0, 0.0, 0.0]), np.array([32.0, 32.0, 32.0])
    )[np.newaxis]  # (1, 8, 3)

    frustum_corners = make_axis_frustum(
        near_center=np.array([-10.0, 16.0, 16.0]), half_size=16.0, depth=20.0
    )
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    _, _, primary_axis = positioning_strategy.position_texture(
        view_params, scale_level_identity, texture_config_64, frustum_chunk_corners
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
    frustum_chunk_corners = chunk_corners_from_bbox(
        np.array([0.0, 0.0, 0.0]), np.array([32.0, 32.0, 32.0])
    )[np.newaxis]

    frustum_corners = make_axis_frustum(
        np.array([16.0, 16.0, 16.0]), half_size=8.0, depth=10.0
    )
    view_params = make_view_params(frustum_corners, view_dir)

    _, _, primary_axis = positioning_strategy.position_texture(
        view_params, scale_level_identity, texture_config_64, frustum_chunk_corners
    )
    assert primary_axis == expected_axis


def test_texture_bounds_are_cube(
    positioning_strategy, scale_level_identity, texture_config_64
):
    """T-03: texture_max - texture_min equals texture_width on every axis."""
    frustum_chunk_corners = chunk_corners_from_bbox(
        np.array([0.0, 0.0, 0.0]), np.array([32.0, 32.0, 32.0])
    )[np.newaxis]

    frustum_corners = make_axis_frustum(
        np.array([-10.0, 16.0, 16.0]), half_size=16.0, depth=20.0
    )
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    _, (tex_min, tex_max), _ = positioning_strategy.position_texture(
        view_params, scale_level_identity, texture_config_64, frustum_chunk_corners
    )
    np.testing.assert_allclose(tex_max - tex_min, texture_config_64.texture_width)


def test_transform_is_pure_translation(
    positioning_strategy, scale_level_identity, texture_config_64
):
    """T-04: Returned AffineTransform has identity upper-left 3x3 block."""
    frustum_chunk_corners = chunk_corners_from_bbox(
        np.array([0.0, 0.0, 0.0]), np.array([32.0, 32.0, 32.0])
    )[np.newaxis]

    frustum_corners = make_axis_frustum(
        np.array([-10.0, 16.0, 16.0]), half_size=16.0, depth=20.0
    )
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    transform, _, _ = positioning_strategy.position_texture(
        view_params, scale_level_identity, texture_config_64, frustum_chunk_corners
    )
    np.testing.assert_allclose(transform.matrix[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(transform.matrix[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6)


def test_transform_translation_equals_texture_min(
    positioning_strategy, scale_level_identity, texture_config_64
):
    """T-05: Mapping texture-space origin (0,0,0) through transform gives tex_min."""
    frustum_chunk_corners = chunk_corners_from_bbox(
        np.array([16.0, 16.0, 16.0]), np.array([48.0, 48.0, 48.0])
    )[np.newaxis]

    frustum_corners = make_axis_frustum(
        np.array([-10.0, 32.0, 32.0]), half_size=8.0, depth=10.0
    )
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    transform, (tex_min, _), _ = positioning_strategy.position_texture(
        view_params, scale_level_identity, texture_config_64, frustum_chunk_corners
    )

    origin = np.array([[0.0, 0.0, 0.0]])
    mapped = transform.map_coordinates(origin)[0]
    np.testing.assert_allclose(mapped, tex_min, atol=1e-6)


def test_camera_low_z_anchors_to_low_z_face_no_clip(
    positioning_strategy, scale_level_identity, texture_config_64
):
    """T-06: Camera on low-z side, texture anchored to low-z face (no clipping)."""
    # 2 chunks along z, 1 along y and x  ->  bbox [0,0,0] to [64,32,32]
    chunks = np.stack(
        [
            chunk_corners_from_bbox(
                np.array([0.0, 0.0, 0.0]), np.array([32.0, 32.0, 32.0])
            ),
            chunk_corners_from_bbox(
                np.array([32.0, 0.0, 0.0]), np.array([64.0, 32.0, 32.0])
            ),
        ],
        axis=0,
    )  # (2, 8, 3)

    frustum_corners = make_axis_frustum(
        near_center=np.array([-100.0, 0.0, 0.0]), half_size=8.0, depth=10.0
    )
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    _, (tex_min, tex_max), _ = positioning_strategy.position_texture(
        view_params, scale_level_identity, texture_config_64, chunks
    )

    np.testing.assert_allclose(tex_min, [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(tex_max, [64.0, 64.0, 64.0], atol=1e-6)


def test_camera_high_z_clips_to_low_z_end(
    positioning_strategy, scale_level_identity, texture_config_64
):
    """T-07: Camera on high-z side, texture still anchors to low-z (bbox_min) end.

    Texture position is determined by bbox_min, not camera orientation, so the
    anchor is always the low-coordinate corner of the frustum AABB regardless
    of which way the camera faces.
    """
    # 4 chunks along z (total 128), 1 along y and x
    chunks = np.stack(
        [
            chunk_corners_from_bbox(
                np.array([i * 32.0, 0.0, 0.0]),
                np.array([(i + 1) * 32.0, 32.0, 32.0]),
            )
            for i in range(4)
        ],
        axis=0,
    )  # (4, 8, 3)

    frustum_corners = make_axis_frustum(
        near_center=np.array([200.0, 0.0, 0.0]), half_size=8.0, depth=10.0
    )
    # Camera looking toward -z (into the scene)
    view_params = make_view_params(frustum_corners, np.array([-1.0, 0.0, 0.0]))

    _, (tex_min, tex_max), _ = positioning_strategy.position_texture(
        view_params, scale_level_identity, texture_config_64, chunks
    )

    np.testing.assert_allclose(tex_min[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(tex_max[0], 64.0, atol=1e-6)


def test_all_axes_clipped_when_oversized(
    positioning_strategy, scale_level_identity, texture_config_64
):
    """T-08: All three axes clipped when AABB greatly exceeds texture_width."""
    chunks = chunk_corners_from_bbox(
        np.array([0.0, 0.0, 0.0]), np.array([200.0, 200.0, 200.0])
    )[np.newaxis]  # pretend it is one giant chunk for positioning purposes

    frustum_corners = make_axis_frustum(
        near_center=np.array([-100.0, 0.0, 0.0]), half_size=8.0, depth=10.0
    )
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    _, (tex_min, tex_max), _ = positioning_strategy.position_texture(
        view_params, scale_level_identity, texture_config_64, chunks
    )

    np.testing.assert_allclose(tex_max - tex_min, 64.0, atol=1e-6)
    np.testing.assert_allclose(tex_min, [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(tex_max, [64.0, 64.0, 64.0], atol=1e-6)


def test_single_chunk_contained_in_texture_bounds(
    positioning_strategy, scale_level_identity, texture_config_64
):
    """T-09: A single chunk's 8 corners fall within texture bounds."""
    single_chunk = chunk_corners_from_bbox(
        np.array([32.0, 32.0, 32.0]), np.array([64.0, 64.0, 64.0])
    )[np.newaxis]  # (1, 8, 3)

    frustum_corners = make_axis_frustum(
        near_center=np.array([-10.0, 48.0, 48.0]), half_size=16.0, depth=20.0
    )
    view_params = make_view_params(frustum_corners, np.array([1.0, 0.0, 0.0]))

    _, (tex_min, tex_max), _ = positioning_strategy.position_texture(
        view_params, scale_level_identity, texture_config_64, single_chunk
    )

    corners = single_chunk[0]  # (8, 3)
    within = np.all(
        (corners >= tex_min[np.newaxis] - 1e-5)
        & (corners <= tex_max[np.newaxis] + 1e-5)
    )
    assert within, (
        f"Single chunk corners not inside texture bounds.\n"
        f"tex=[{tex_min}, {tex_max}]\nchunk corners=\n{corners}"
    )


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
        scale_level_identity, view_params, texture_config_64
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
        scale_level_identity, view_params, texture_config_64
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
        scale_level_identity, view_params, texture_config_64
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
        scale_level_identity, view_params, TextureConfiguration(texture_width=32)
    )
    result_large = chunk_selector.select_chunks(
        scale_level_identity, view_params, TextureConfiguration(texture_width=128)
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
