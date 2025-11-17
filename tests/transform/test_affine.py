"""Tests for the AffineTransform class."""

import numpy as np

from cellier.transform import AffineTransform


def test_coordinate_transform_combined_scale_translation():
    """Test coordinate transforms with combined scale and translation."""
    # Create transform with non-uniform scale and translation
    scale = (2.0, 0.5, 3.0)
    translation = (10.0, -5.0, 7.0)
    transform = AffineTransform.from_scale_and_translation(scale, translation)

    # check the matrix
    expected_transform = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 0.5, 0.0, -5.0],
            [0.0, 0.0, 3.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(transform.matrix, expected_transform)

    # Test coordinates (batch of 4 points)
    coordinates = np.array(
        [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-1.0, 4.0, -2.0], [5.0, -3.0, 1.0]]
    )

    # Expected results: scale then translate
    expected_transformed = np.array(
        [
            [
                1.0 * 2.0 + 10.0,
                2.0 * 0.5 + (-5.0),
                3.0 * 3.0 + 7.0,
            ],  # [12.0, -4.0, 16.0]
            [
                0.0 * 2.0 + 10.0,
                0.0 * 0.5 + (-5.0),
                0.0 * 3.0 + 7.0,
            ],  # [10.0, -5.0, 7.0]
            [
                -1.0 * 2.0 + 10.0,
                4.0 * 0.5 + (-5.0),
                -2.0 * 3.0 + 7.0,
            ],  # [8.0, -3.0, 1.0]
            [
                5.0 * 2.0 + 10.0,
                -3.0 * 0.5 + (-5.0),
                1.0 * 3.0 + 7.0,
            ],  # [20.0, -6.5, 10.0]
        ]
    )

    # Test forward transform
    transformed = transform.map_coordinates(coordinates)
    np.testing.assert_allclose(transformed, expected_transformed, rtol=1e-6)

    # Test inverse transform
    inverse_transformed = transform.imap_coordinates(transformed)
    np.testing.assert_allclose(inverse_transformed, coordinates, atol=1e-6)


def test_coordinate_transform_roundtrip_accuracy():
    """Test that forward → inverse transform returns original coordinates."""
    # More complex transform matrix
    transform = AffineTransform.from_scale_and_translation(
        scale=(1.5, 2.5, 0.8), translation=(3.0, -7.0, 12.0)
    )

    # Batch of test coordinates
    original_coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [-5.0, 3.0, 2.0],
            [10.0, -4.0, 8.0],
            [2.5, -1.5, 0.0],
        ]
    )

    # Forward then inverse transform
    transformed = transform.map_coordinates(original_coordinates)
    roundtrip = transform.imap_coordinates(transformed)

    np.testing.assert_allclose(roundtrip, original_coordinates, atol=1e-6)


def test_normal_vector_transform_with_scale_translation():
    """Test normal vector transforms with non-uniform scaling and translation."""
    # Non-uniform scale should affect normals, translation should not
    scale = (2.0, 4.0, 0.5)
    translation = (10.0, -5.0, 7.0)
    transform = AffineTransform.from_scale_and_translation(scale, translation)

    # Batch of unit normal vectors
    normals = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0] / np.sqrt(2),  # normalized diagonal
            [1.0, 1.0, 1.0] / np.sqrt(3),  # normalized diagonal
        ]
    )

    # Transform normals
    transformed_normals = transform.map_normal_vector(normals)

    # Check that all transformed normals are unit vectors
    norms = np.linalg.norm(transformed_normals, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-6)

    i_transformed_normals = transform.imap_normal_vector(transformed_normals)
    np.testing.assert_allclose(i_transformed_normals, normals, rtol=1e-6)


def test_normal_vector_transform_roundtrip_accuracy():
    """Test that normal → inverse normal transform returns original vectors."""
    transform = AffineTransform.from_scale_and_translation(
        scale=(3.0, 1.5, 2.0), translation=(1.0, 2.0, 3.0)
    )

    # Batch of original normal vectors (already normalized)
    original_normals = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0] / np.sqrt(2),
            [1.0, 0.0, 1.0] / np.sqrt(2),
            [0.0, 1.0, 1.0] / np.sqrt(2),
        ]
    )

    # Forward then inverse transform
    transformed = transform.map_normal_vector(original_normals)
    roundtrip = transform.imap_normal_vector(transformed)

    np.testing.assert_allclose(roundtrip, original_normals, atol=1e-6)


def test_normal_vector_unit_preservation():
    """Test that transformed normals are always unit vectors."""
    transform = AffineTransform.from_scale_and_translation(
        scale=(0.1, 10.0, 5.0),  # Extreme scaling
        translation=(100.0, -200.0, 50.0),
    )

    # Various normal vectors
    # (some not initially unit vectors, but will be normalized by to_vec4)
    normals = np.array(
        [
            [2.0, 0.0, 0.0],  # Will be normalized
            [0.0, 3.0, 0.0],  # Will be normalized
            [1.0, 1.0, 1.0],  # Will be normalized
            [1.0, 2.0, 3.0],  # Will be normalized
            [0.5, 0.5, 0.5],  # Will be normalized
        ]
    )

    # Normalize input normals first (since the method expects unit vectors)
    normalized_normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # Transform normals
    transformed = transform.map_normal_vector(normalized_normals)

    # Check all are unit vectors
    norms = np.linalg.norm(transformed, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_from_translation_constructor():
    """Test the from_translation constructor."""
    translation = (5.0, -3.0, 8.0)
    transform = AffineTransform.from_translation(translation)

    # Test that the matrix is correct
    expected_matrix = np.eye(4, dtype=np.float32)
    expected_matrix[0, 3] = 5.0
    expected_matrix[1, 3] = -3.0
    expected_matrix[2, 3] = 8.0

    np.testing.assert_allclose(transform.matrix, expected_matrix)

    # Test coordinate mapping
    coordinates = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])

    expected_transformed = coordinates + np.array([5.0, -3.0, 8.0])
    transformed = transform.map_coordinates(coordinates)

    np.testing.assert_allclose(transformed, expected_transformed, atol=1e-6)


def test_from_scale_and_translation_constructor():
    """Test the from_scale_and_translation constructor."""
    scale = (2.0, 0.5, 3.0)
    translation = (1.0, -2.0, 4.0)
    transform = AffineTransform.from_scale_and_translation(scale, translation)

    # Test that the matrix is correct
    expected_matrix = np.eye(4, dtype=np.float32)
    expected_matrix[0, 0] = 2.0
    expected_matrix[1, 1] = 0.5
    expected_matrix[2, 2] = 3.0
    expected_matrix[0, 3] = 1.0
    expected_matrix[1, 3] = -2.0
    expected_matrix[2, 3] = 4.0

    np.testing.assert_allclose(transform.matrix, expected_matrix)

    # Test coordinate mapping
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 1.0], [2.0, 4.0, -1.0], [-1.0, -2.0, 0.5]]
    )

    # Expected: scale then translate
    expected_transformed = np.array(
        [
            [0.0 * 2.0 + 1.0, 0.0 * 0.5 + (-2.0), 0.0 * 3.0 + 4.0],  # [1.0, -2.0, 4.0]
            [1.0 * 2.0 + 1.0, 2.0 * 0.5 + (-2.0), 1.0 * 3.0 + 4.0],  # [3.0, -1.0, 7.0]
            [2.0 * 2.0 + 1.0, 4.0 * 0.5 + (-2.0), -1.0 * 3.0 + 4.0],  # [5.0, 0.0, 1.0]
            [
                -1.0 * 2.0 + 1.0,
                -2.0 * 0.5 + (-2.0),
                0.5 * 3.0 + 4.0,
            ],  # [-1.0, -3.0, 5.5]
        ]
    )

    transformed = transform.map_coordinates(coordinates)
    np.testing.assert_allclose(transformed, expected_transformed, atol=1e-6)
