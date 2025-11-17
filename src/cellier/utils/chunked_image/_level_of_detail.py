"""Functions for selecting the level of detail for a chunked image."""

import numpy as np

from cellier.utils.chunked_image._multiscale_image_model import (
    MultiscaleImageModel,
)


def select_scale_from_view(
    image_model: MultiscaleImageModel,
    frustum_corners: np.ndarray,
    viewport_size: tuple[int, int],
    max_texture_sizes: dict[int, tuple[int, int, int]],
    quality_factor: float = 1.0,
    target_chunk_pixels: float = 1.0,
) -> int:
    """Select the optimal scale level for chunked rendering.

    This function balances visual quality with performance by selecting a scale
    level where the ratio of chunk resolution to screen-space size approaches
    the target ratio, while respecting texture size constraints.

    Parameters
    ----------
    image_model : MultiscaleImageModel
        The multiscale image model containing all scale levels.
    frustum_corners : np.ndarray
        View frustum corners in world coordinates, shape (2, 4, 3).
        First axis: near/far planes, second: corners within plane,
        third: (z, y, x) coordinates.
    viewport_size : tuple[int, int]
        Viewport dimensions in pixels (width, height).
    max_texture_sizes : dict[int, tuple[int, int, int]]
        Maximum texture size for each scale level.
        Key: scale_index, Value: (width, height, depth) in pixels.
    quality_factor : float, optional
        Quality bias factor. Default is 1.0.
        < 1.0 prefers lower resolution (better performance).
        > 1.0 prefers higher resolution (better quality).
    target_chunk_pixels : float, optional
        Target ratio of chunk pixels to screen pixels per chunk dimension.
        Default is 1.0 (one chunk pixel per screen pixel).
        Values > 1.0 create oversampling, < 1.0 create undersampling.

    Returns
    -------
    int
        Index of the selected scale level (0 is finest resolution).

    Raises
    ------
    ValueError
        If no scale level meets the texture size constraint.
    """
    viewport_pixels = viewport_size[0] * viewport_size[1]
    adjusted_target = target_chunk_pixels * quality_factor

    # Filter scales that meet texture size constraint
    valid_scales = []
    required_coverage = max(viewport_size)  # Must cover full screen dimension

    for scale_idx in range(image_model.n_scales):
        if scale_idx in max_texture_sizes:
            texture_cube_size = max_texture_sizes[scale_idx][0]  # Assume cube texture

            # Calculate actual screen-space coverage of texture cube
            texture_coverage = _calculate_texture_screen_coverage(
                image_model.get_scale(scale_idx),
                texture_cube_size,
                frustum_corners,
                viewport_size,
            )

            if texture_coverage >= required_coverage:
                valid_scales.append(scale_idx)

    if not valid_scales:
        raise ValueError(
            f"No scale level meets texture size constraint. "
            f"Viewport requires {viewport_pixels} pixels, "
            f"but maximum available texture sizes are: {max_texture_sizes}"
        )

    # Calculate suitability score for each valid scale
    best_scale = valid_scales[0]  # Fallback to highest resolution valid scale
    best_score = float("inf")

    for scale_idx in valid_scales:
        scale_level = image_model.get_scale(scale_idx)

        # Calculate representative chunk screen-space size
        chunk_screen_size = _calculate_chunk_screen_size(
            scale_level, frustum_corners, viewport_size
        )

        # Apply resolution ceiling based on chunk capabilities
        max_chunk_resolution = max(scale_level.chunk_shape)

        # Calculate target ratio: chunk_pixels / screen_pixels
        # adjusted_target is our desired ratio
        if chunk_screen_size > 0:
            actual_ratio = max_chunk_resolution / chunk_screen_size
            # Calculate how well this scale matches our target ratio
            target_ratio_error = abs(adjusted_target - actual_ratio)

            if target_ratio_error < best_score:
                best_score = target_ratio_error
                best_scale = scale_idx

    return best_scale


def _calculate_texture_screen_coverage(
    scale_level,
    texture_cube_size: float,
    frustum_corners: np.ndarray,
    viewport_size: tuple[int, int],
) -> float:
    """Calculate the screen-space coverage of a texture cube at a given scale.

    This function:
    1. Converts the frustum to scale coordinates
    2. Positions a cube texture at the frustum center in scale coordinates
    3. Projects the texture cube corners to the average frustum plane
    4. Calculates the maximum screen-space dimension of the projected cube

    Parameters
    ----------
    scale_level : ScaleLevel
        The scale level to analyze.
    texture_cube_size : float
        Side length of the cube texture in scale coordinates.
    frustum_corners : np.ndarray
        View frustum corners in world coordinates, shape (2, 4, 3).
    viewport_size : tuple[int, int]
        Viewport dimensions (width, height).

    Returns
    -------
    float
        Maximum screen-space dimension of the texture cube in pixels.
    """
    # Convert frustum from world coordinates to scale coordinates
    frustum_scale = _transform_frustum_to_scale(frustum_corners, scale_level)

    # Calculate frustum center in scale coordinates
    frustum_center = np.mean(frustum_scale.reshape(-1, 3), axis=0)

    # Position texture cube at frustum center
    half_size = texture_cube_size / 2
    cube_corners = np.array(
        [
            frustum_center + np.array([-half_size, -half_size, -half_size]),
            frustum_center + np.array([+half_size, -half_size, -half_size]),
            frustum_center + np.array([-half_size, +half_size, -half_size]),
            frustum_center + np.array([+half_size, +half_size, -half_size]),
            frustum_center + np.array([-half_size, -half_size, +half_size]),
            frustum_center + np.array([+half_size, -half_size, +half_size]),
            frustum_center + np.array([-half_size, +half_size, +half_size]),
            frustum_center + np.array([+half_size, +half_size, +half_size]),
        ]
    )

    # Project cube corners to average frustum plane
    average_plane = (
        frustum_scale[0] + frustum_scale[1]
    ) / 2  # Average of near and far planes
    projected_points = _project_points_to_plane(cube_corners, average_plane)

    # Calculate screen-space coverage
    return _calculate_screen_coverage_from_projected_points(
        projected_points, frustum_scale, viewport_size
    )


def _transform_frustum_to_scale(frustum_corners: np.ndarray, scale_level) -> np.ndarray:
    """Transform frustum corners from world coordinates to scale coordinates.

    Parameters
    ----------
    frustum_corners : np.ndarray
        Frustum corners in world coordinates, shape (2, 4, 3).
    scale_level : ScaleLevel
        The scale level containing the transformation.

    Returns
    -------
    np.ndarray
        Frustum corners in scale coordinates, same shape as input.
    """
    # Flatten to apply transformation
    corners_flat = frustum_corners.reshape(-1, 3)

    # Transform from world to scale coordinates (inverse of scale transform)
    corners_scale_flat = scale_level.transform.imap_coordinates(corners_flat)

    # Reshape back to original frustum shape
    return corners_scale_flat.reshape(2, 4, 3)


def _project_points_to_plane(
    points: np.ndarray, plane_points: np.ndarray
) -> np.ndarray:
    """Project 3D points onto a plane defined by 4 corner points.

    This uses the plane defined by the 4 corner points and projects
    each input point onto that plane using the plane's normal vector.

    Parameters
    ----------
    points : np.ndarray
        Points to project, shape (n_points, 3).
    plane_points : np.ndarray
        Four corner points defining the plane, shape (4, 3).

    Returns
    -------
    np.ndarray
        Projected points in the plane's 2D coordinate system, shape (n_points, 2).
    """
    # Calculate plane normal using cross product of two edges
    edge1 = plane_points[1] - plane_points[0]
    edge2 = plane_points[3] - plane_points[0]
    normal = np.cross(edge1, edge2)
    normal = normal / np.linalg.norm(normal)

    # Project points onto the plane
    plane_origin = plane_points[0]
    projected_3d = []

    for point in points:
        # Vector from plane origin to point
        vec_to_point = point - plane_origin
        # Project onto plane by removing component along normal
        projection = vec_to_point - np.dot(vec_to_point, normal) * normal
        projected_3d.append(plane_origin + projection)

    projected_3d = np.array(projected_3d)

    # Convert 3D projected points to 2D plane coordinates
    # Use two edges of the plane as basis vectors
    u_axis = edge1 / np.linalg.norm(edge1)
    v_axis = edge2 / np.linalg.norm(edge2)

    projected_2d = []
    for point_3d in projected_3d:
        relative_vec = point_3d - plane_origin
        u_coord = np.dot(relative_vec, u_axis)
        v_coord = np.dot(relative_vec, v_axis)
        projected_2d.append([u_coord, v_coord])

    return np.array(projected_2d)


def _calculate_screen_coverage_from_projected_points(
    projected_points: np.ndarray,
    frustum_scale: np.ndarray,
    viewport_size: tuple[int, int],
) -> float:
    """Calculate screen-space coverage from 2D projected points.

    Parameters
    ----------
    projected_points : np.ndarray
        2D projected points, shape (n_points, 2).
    frustum_scale : np.ndarray
        Frustum corners in scale coordinates, shape (2, 4, 3).
    viewport_size : tuple[int, int]
        Viewport dimensions (width, height).

    Returns
    -------
    float
        Maximum screen-space dimension in pixels.
    """
    if len(projected_points) == 0:
        return 0.0

    # Calculate bounding box of projected points
    min_coords = np.min(projected_points, axis=0)
    max_coords = np.max(projected_points, axis=0)
    projection_width = max_coords[0] - min_coords[0]
    projection_height = max_coords[1] - min_coords[1]

    # Calculate frustum dimensions at the average plane
    average_plane = (frustum_scale[0] + frustum_scale[1]) / 2
    frustum_width = np.linalg.norm(average_plane[1] - average_plane[0])  # Right - left
    frustum_height = np.linalg.norm(average_plane[3] - average_plane[0])  # Top - bottom

    # Convert to screen pixels using linear scaling
    if frustum_width > 0 and frustum_height > 0:
        screen_width = (projection_width / frustum_width) * viewport_size[0]
        screen_height = (projection_height / frustum_height) * viewport_size[1]
        return max(screen_width, screen_height)

    return 0.0


def _calculate_chunk_screen_size(
    scale_level, frustum_corners: np.ndarray, viewport_size: tuple[int, int]
) -> float:
    """Calculate representative chunk screen-space size for a scale level.

    This function samples a few representative chunks to estimate typical
    screen-space coverage, prioritizing performance over perfect accuracy.

    Parameters
    ----------
    scale_level : ScaleLevel
        The scale level to analyze.
    frustum_corners : np.ndarray
        View frustum corners in world coordinates.
    viewport_size : Tuple[int, int]
        Viewport dimensions (width, height).

    Returns
    -------
    float
        Representative chunk size in screen pixels.
    """
    # For performance, sample only a few representative chunks
    n_chunks = scale_level.n_chunks
    if n_chunks == 0:
        return 0.0

    # Sample strategy: take a few chunks distributed through the dataset
    sample_indices = _get_representative_chunk_indices(n_chunks, max_samples=5)

    chunk_sizes = []
    for chunk_idx in sample_indices:
        # Get chunk corners in world coordinates (scale_0)
        chunk_corners_world = scale_level.chunk_corners_scale_0[chunk_idx]

        # Calculate screen-space size
        screen_size = _project_chunk_to_screen(
            chunk_corners_world, frustum_corners, viewport_size
        )

        if screen_size > 0:
            chunk_sizes.append(screen_size)

    # Return median size for robustness (avoids outliers from edge chunks)
    if chunk_sizes:
        return float(np.median(chunk_sizes))
    else:
        return 0.0


def _get_representative_chunk_indices(n_chunks: int, max_samples: int = 5) -> list[int]:
    """Get indices of representative chunks for sampling.

    Parameters
    ----------
    n_chunks : int
        Total number of chunks.
    max_samples : int
        Maximum number of chunks to sample.

    Returns
    -------
    list[int]
        List of chunk indices to sample.
    """
    if n_chunks <= max_samples:
        return list(range(n_chunks))

    # Sample chunks distributed throughout the dataset
    indices = np.linspace(0, n_chunks - 1, max_samples, dtype=int)
    return indices.tolist()


def _project_chunk_to_screen(
    chunk_corners_world: np.ndarray,
    frustum_corners: np.ndarray,
    viewport_size: tuple[int, int],
) -> float:
    """Project chunk corners to screen space and calculate coverage.

    This uses a simplified projection that estimates screen-space size
    without full camera matrix calculations for performance. The approach:

    1. Calculates the world-space dimensions of the viewing frustum at the near plane
    2. Calculates the world-space dimensions of the chunk bounding box
    3. Estimates screen coverage by assuming linear scaling:
       screen_pixels = (chunk_size / frustum_size) * viewport_pixels

    This approximation works well for chunks that are roughly at the same depth
    as the near plane, but may be less accurate for chunks at very different
    depths or with significant perspective distortion.

    Parameters
    ----------
    chunk_corners_world : np.ndarray
        Chunk corners in world coordinates, shape (8, 3).
    frustum_corners : np.ndarray
        View frustum corners, shape (2, 4, 3).
    viewport_size : tuple[int, int]
        Viewport dimensions (width, height).

    Returns
    -------
    float
        Estimated chunk size in screen pixels.
    """
    # Simplified projection: estimate based on frustum geometry
    # This is less accurate than full matrix projection but much faster

    # Calculate frustum dimensions at the near plane
    near_plane = frustum_corners[0]  # Shape (4, 3)
    frustum_width = np.linalg.norm(near_plane[1] - near_plane[0])  # Right - left
    frustum_height = np.linalg.norm(near_plane[3] - near_plane[0])  # Top - bottom

    # Calculate chunk dimensions in world space
    chunk_min = np.min(chunk_corners_world, axis=0)
    chunk_max = np.max(chunk_corners_world, axis=0)
    chunk_width = chunk_max[2] - chunk_min[2]  # x dimension
    chunk_height = chunk_max[1] - chunk_min[1]  # y dimension

    # Estimate screen-space size (simplified perspective projection)
    if frustum_width > 0 and frustum_height > 0:
        screen_width = (chunk_width / frustum_width) * viewport_size[0]
        screen_height = (chunk_height / frustum_height) * viewport_size[1]

        # Return the larger dimension in pixels
        return max(screen_width, screen_height)

    return 0.0
