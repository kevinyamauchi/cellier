import numpy as np
import pytest
from pydantic import ValidationError

from cellier.v2.transform import AffineTransform

# ---------------------------------------------------------------------------
# 3D tests (backwards compatibility)
# ---------------------------------------------------------------------------


def test_identity_maps_point_to_itself():
    t = AffineTransform.identity()
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(t.map_coordinates(pts), pts, atol=1e-6)
    np.testing.assert_allclose(t.imap_coordinates(pts), pts, atol=1e-6)


def test_identity_json_roundtrip():
    t = AffineTransform.identity()
    json_str = t.model_dump_json()
    t2 = AffineTransform.model_validate_json(json_str)
    np.testing.assert_array_equal(t.matrix, t2.matrix)


def test_from_scale_diagonal():
    t = AffineTransform.from_scale((2.0, 3.0, 4.0))
    expected = np.diag([2.0, 3.0, 4.0, 1.0]).astype(np.float32)
    np.testing.assert_array_equal(t.matrix, expected)

    pts = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    result = t.map_coordinates(pts)
    np.testing.assert_allclose(result, [[2.0, 3.0, 4.0]], atol=1e-6)

    recovered = t.imap_coordinates(result)
    np.testing.assert_allclose(recovered, pts, atol=1e-6)


def test_from_translation_constructor():
    t = AffineTransform.from_translation((10.0, 20.0, 30.0))
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    result = t.map_coordinates(pts)
    np.testing.assert_allclose(result, [[10.0, 20.0, 30.0]], atol=1e-6)

    json_str = t.model_dump_json()
    t2 = AffineTransform.model_validate_json(json_str)
    np.testing.assert_allclose(t2.matrix, t.matrix, atol=1e-6)


def test_from_scale_and_translation_constructor():
    t = AffineTransform.from_scale_and_translation((2.0, 2.0, 2.0), (5.0, 5.0, 5.0))
    pts = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    result = t.map_coordinates(pts)
    np.testing.assert_allclose(result, [[7.0, 7.0, 7.0]], atol=1e-6)

    json_str = t.model_dump_json()
    t2 = AffineTransform.model_validate_json(json_str)
    np.testing.assert_allclose(t2.matrix, t.matrix, atol=1e-6)


def test_map_coordinates_roundtrip():
    t = AffineTransform.from_scale_and_translation((2.0, 0.5, 3.0), (10.0, -5.0, 7.0))
    pts = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    recovered = t.imap_coordinates(t.map_coordinates(pts))
    np.testing.assert_allclose(recovered, pts, atol=1e-5)


def test_invalid_matrix_shape_raises():
    with pytest.raises(ValueError, match="square"):
        AffineTransform(matrix=np.ones((3, 4)))


def test_frozen_raises_on_mutation():
    t = AffineTransform.identity()
    with pytest.raises(ValidationError):
        t.matrix = np.eye(4)


def test_inverse_matrix_cached():
    t = AffineTransform.from_scale_and_translation((2.0, 3.0, 4.0), (1.0, 2.0, 3.0))
    expected = np.linalg.inv(t.matrix).astype(np.float32)
    np.testing.assert_allclose(t.inverse_matrix, expected, atol=1e-6)
    # Verify it is the same object (cached).
    assert t.inverse_matrix is t.inverse_matrix


def test_normal_vector_transform_known_result():
    # Non-uniform scale: (2, 1, 1). A normal along (1, 0, 0) in data
    # space should map to (0.5, 0, 0) normalised = (1, 0, 0) in world.
    # But a normal along (1, 1, 0) / sqrt(2) should NOT simply scale.
    t = AffineTransform.from_scale((2.0, 1.0, 1.0))

    # Normal along x in data space.
    n = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    result = t.map_normal_vector(n)
    np.testing.assert_allclose(result, [[1.0, 0.0, 0.0]], atol=1e-6)

    # Diagonal normal — should be skewed by the inverse-transpose.
    n_diag = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
    n_diag = n_diag / np.linalg.norm(n_diag)
    mapped = t.map_normal_vector(n_diag)
    # The mapped normal should not be the same as simply scaling.
    scaled = n_diag * np.array([[2.0, 1.0, 1.0]])
    scaled = scaled / np.linalg.norm(scaled)
    assert not np.allclose(mapped, scaled, atol=1e-3)
    # But it should still be unit length.
    np.testing.assert_allclose(np.linalg.norm(mapped), 1.0, atol=1e-6)


def test_normal_vector_roundtrip():
    t = AffineTransform.from_scale((2.0, 3.0, 4.0))
    n = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    mapped = t.map_normal_vector(n)
    recovered = t.imap_normal_vector(mapped)
    # Directions should match (up to sign and normalisation).
    np.testing.assert_allclose(np.abs(recovered), np.abs(n), atol=1e-5)


def test_compose_order():
    scale = AffineTransform.from_scale((2.0, 2.0, 2.0))
    translate = AffineTransform.from_translation((10.0, 0.0, 0.0))

    composed = scale @ translate
    pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    # (scale @ translate)(p) == translate(scale(p)) == translate(2, 0, 0) == (12, 0, 0)
    expected = translate.map_coordinates(scale.map_coordinates(pts))
    result = composed.map_coordinates(pts)
    np.testing.assert_allclose(result, expected, atol=1e-6)
    np.testing.assert_allclose(result, [[12.0, 0.0, 0.0]], atol=1e-6)


def test_compose_noncommutative():
    scale = AffineTransform.from_scale((2.0, 2.0, 2.0))
    translate = AffineTransform.from_translation((10.0, 0.0, 0.0))

    forward = (scale @ translate).map_coordinates(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    )
    reverse = (translate @ scale).map_coordinates(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    )
    assert not np.allclose(forward, reverse)


def test_to_homogeneous_private():
    # _to_homogeneous is a private module function, not exported.
    from cellier.v2.transform import __all__

    assert "_to_homogeneous" not in __all__


def test_ndim_property():
    assert AffineTransform.identity().ndim == 3
    assert AffineTransform.identity(ndim=4).ndim == 4
    assert AffineTransform.from_scale((1.0, 2.0)).ndim == 2
    assert AffineTransform.from_scale((1.0, 2.0, 3.0, 4.0, 5.0)).ndim == 5


# ---------------------------------------------------------------------------
# N-D tests
# ---------------------------------------------------------------------------


def test_identity_4d():
    t = AffineTransform.identity(ndim=4)
    assert t.matrix.shape == (5, 5)
    pts = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(t.map_coordinates(pts), pts, atol=1e-6)


def test_from_scale_4d():
    t = AffineTransform.from_scale((1.0, 4.0, 2.0, 2.0))
    assert t.ndim == 4
    assert t.matrix.shape == (5, 5)
    expected_diag = np.diag([1.0, 4.0, 2.0, 2.0, 1.0]).astype(np.float32)
    np.testing.assert_array_equal(t.matrix, expected_diag)

    pts = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    result = t.map_coordinates(pts)
    np.testing.assert_allclose(result, [[1.0, 4.0, 2.0, 2.0]], atol=1e-6)

    recovered = t.imap_coordinates(result)
    np.testing.assert_allclose(recovered, pts, atol=1e-6)


def test_from_scale_and_translation_4d():
    t = AffineTransform.from_scale_and_translation(
        (1.0, 4.0, 2.0, 2.0), (0.0, 10.0, 5.0, 5.0)
    )
    pts = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    result = t.map_coordinates(pts)
    np.testing.assert_allclose(result, [[0.0, 10.0, 5.0, 5.0]], atol=1e-6)


def test_from_translation_4d():
    t = AffineTransform.from_translation((0.0, 10.0, 5.0, 5.0))
    assert t.ndim == 4
    pts = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    result = t.map_coordinates(pts)
    np.testing.assert_allclose(result, [[1.0, 12.0, 8.0, 9.0]], atol=1e-6)


def test_identity_4d_json_roundtrip():
    t = AffineTransform.identity(ndim=4)
    json_str = t.model_dump_json()
    t2 = AffineTransform.model_validate_json(json_str)
    np.testing.assert_array_equal(t.matrix, t2.matrix)
    assert t2.ndim == 4


def test_map_coordinates_roundtrip_4d():
    t = AffineTransform.from_scale_and_translation(
        (1.0, 2.0, 0.5, 3.0), (0.0, 10.0, -5.0, 7.0)
    )
    pts = np.array([[0.0, 1.0, 2.0, 3.0], [5.0, 4.0, 5.0, 6.0]], dtype=np.float32)
    recovered = t.imap_coordinates(t.map_coordinates(pts))
    np.testing.assert_allclose(recovered, pts, atol=1e-5)


def test_compose_ndim_mismatch_raises():
    t3 = AffineTransform.identity(ndim=3)
    t4 = AffineTransform.identity(ndim=4)
    with pytest.raises(ValueError, match="different ndim"):
        t3 @ t4


def test_compose_4d():
    scale = AffineTransform.from_scale((1.0, 2.0, 2.0, 2.0))
    translate = AffineTransform.from_translation((0.0, 10.0, 0.0, 0.0))
    composed = scale @ translate
    pts = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    result = composed.map_coordinates(pts)
    np.testing.assert_allclose(result, [[0.0, 12.0, 0.0, 0.0]], atol=1e-6)


def test_from_scale_2d():
    t = AffineTransform.from_scale((3.0, 5.0))
    assert t.ndim == 2
    assert t.matrix.shape == (3, 3)
    pts = np.array([[2.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(t.map_coordinates(pts), [[6.0, 20.0]], atol=1e-6)


def test_scale_translation_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        AffineTransform.from_scale_and_translation((1.0, 2.0, 3.0), (1.0, 2.0))


# ---------------------------------------------------------------------------
# set_slice / expand_dims tests
# ---------------------------------------------------------------------------


def test_set_slice_3d_to_2d():
    """Extract axes 1,2 (y,x) from a 3D transform."""
    t = AffineTransform.from_scale_and_translation((4.0, 2.0, 3.0), (10.0, 20.0, 30.0))
    sub = t.set_slice((1, 2))
    assert sub.ndim == 2
    assert sub.matrix.shape == (3, 3)
    # Scale: y=2, x=3
    np.testing.assert_allclose(sub.matrix[0, 0], 2.0)
    np.testing.assert_allclose(sub.matrix[1, 1], 3.0)
    # Translation: y=20, x=30
    np.testing.assert_allclose(sub.matrix[0, 2], 20.0)
    np.testing.assert_allclose(sub.matrix[1, 2], 30.0)


def test_set_slice_4d_to_3d():
    """Extract last 3 axes from a 4D transform."""
    t = AffineTransform.from_scale((1.0, 4.0, 2.0, 3.0))
    sub = t.set_slice((1, 2, 3))
    assert sub.ndim == 3
    assert sub.matrix.shape == (4, 4)
    expected = np.diag([4.0, 2.0, 3.0, 1.0]).astype(np.float32)
    np.testing.assert_array_equal(sub.matrix, expected)


def test_set_slice_non_contiguous_axes():
    """Extract axes 0,2 (t,y) from a 4D transform, skipping z."""
    t = AffineTransform.from_scale((1.0, 4.0, 2.0, 3.0))
    sub = t.set_slice((0, 2))
    assert sub.ndim == 2
    # Scale: t=1, y=2
    np.testing.assert_allclose(sub.matrix[0, 0], 1.0)
    np.testing.assert_allclose(sub.matrix[1, 1], 2.0)


def test_set_slice_preserves_off_diagonal():
    """Off-diagonal entries (shear/rotation) are preserved in the sub-matrix."""
    m = np.eye(4, dtype=np.float32)
    m[0, 1] = 0.5  # shear between axes 0 and 1
    m[1, 0] = 0.3
    t = AffineTransform(matrix=m)
    sub = t.set_slice((0, 1))
    np.testing.assert_allclose(sub.matrix[0, 1], 0.5)
    np.testing.assert_allclose(sub.matrix[1, 0], 0.3)


def test_set_slice_roundtrip_coordinates():
    """Sliced transform should produce same result as full transform for those axes."""
    t = AffineTransform.from_scale_and_translation(
        (1.0, 4.0, 2.0, 3.0), (0.0, 10.0, 5.0, 7.0)
    )
    sub = t.set_slice((1, 2, 3))

    # Map a 3D point through the sub-transform
    pts_3d = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    result_sub = sub.map_coordinates(pts_3d)

    # Map a 4D point through the full transform, extract last 3
    pts_4d = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
    result_full = t.map_coordinates(pts_4d)

    np.testing.assert_allclose(result_sub, result_full[:, 1:], atol=1e-6)


def test_expand_dims_3d_to_5d():
    t = AffineTransform.from_scale((4.0, 2.0, 3.0))
    expanded = t.expand_dims(5)
    assert expanded.ndim == 5
    assert expanded.matrix.shape == (6, 6)
    # Leading 2 axes are identity
    np.testing.assert_allclose(expanded.matrix[0, 0], 1.0)
    np.testing.assert_allclose(expanded.matrix[1, 1], 1.0)
    # Last 3 axes carry the original transform
    np.testing.assert_allclose(expanded.matrix[2, 2], 4.0)
    np.testing.assert_allclose(expanded.matrix[3, 3], 2.0)
    np.testing.assert_allclose(expanded.matrix[4, 4], 3.0)


def test_expand_dims_with_translation():
    t = AffineTransform.from_scale_and_translation((2.0, 3.0), (10.0, 20.0))
    expanded = t.expand_dims(4)
    assert expanded.ndim == 4
    # Leading 2 axes: identity, no translation
    np.testing.assert_allclose(expanded.matrix[0, 4], 0.0)
    np.testing.assert_allclose(expanded.matrix[1, 4], 0.0)
    # Last 2 axes: scale + translation
    np.testing.assert_allclose(expanded.matrix[2, 2], 2.0)
    np.testing.assert_allclose(expanded.matrix[3, 3], 3.0)
    np.testing.assert_allclose(expanded.matrix[2, 4], 10.0)
    np.testing.assert_allclose(expanded.matrix[3, 4], 20.0)


def test_expand_dims_same_ndim_returns_self():
    t = AffineTransform.from_scale((2.0, 3.0, 4.0))
    expanded = t.expand_dims(3)
    assert expanded is t


def test_expand_dims_smaller_raises():
    t = AffineTransform.from_scale((2.0, 3.0, 4.0))
    with pytest.raises(ValueError, match="target_ndim"):
        t.expand_dims(2)
