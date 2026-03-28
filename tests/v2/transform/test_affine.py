import numpy as np
import pytest
from pydantic import ValidationError

from cellier.v2.transform import AffineTransform


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
    with pytest.raises(ValueError, match=r"shape \(4, 4\)"):
        AffineTransform(matrix=np.eye(3))


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


def test_to_vec4_private():
    # _to_vec4 is a private module function, not exported.
    from cellier.v2.transform import __all__

    assert "_to_vec4" not in __all__
