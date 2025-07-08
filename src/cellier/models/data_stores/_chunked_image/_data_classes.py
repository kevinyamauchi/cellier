"""Data structures for chunk selection algorithms.

This module defines the core data structures used throughout the chunk selection
system for chunked 3D image rendering. These structures encapsulate camera view
information, texture configuration, and algorithm results.
"""

from dataclasses import dataclass

import numpy as np

from cellier.transform import AffineTransform


@dataclass
class ViewParameters:
    """Camera view information for chunk selection algorithms.

    This dataclass encapsulates all the camera and view-related parameters
    needed by chunk selection algorithms to determine optimal texture placement
    and chunk filtering.

    Parameters
    ----------
    frustum_corners : np.ndarray
        Array of shape (2, 4, 3) containing the corners of the view frustum.
        The first dimension represents near (index 0) and far (index 1) planes.
        The second dimension represents the 4 corners of each plane in order:
        (left-bottom, right-bottom, right-top, left-top).
        The third dimension contains (z, y, x) coordinates in world space.
        These corners define the 3D volume visible to the camera.
    view_direction : np.ndarray
        Array of shape (3,) containing the normalized view direction vector
        in world coordinates as (z, y, x). This vector points from the camera
        toward the scene and should have unit length (magnitude = 1.0).
    near_plane_center : np.ndarray
        Array of shape (3,) containing the center point of the near clipping
        plane in world coordinates as (z, y, x). This point is typically used
        as a reference for texture positioning algorithms.

    Notes
    -----
    All coordinate arrays use the (z, y, x) ordering convention to maintain
    consistency with the existing Cellier codebase. The view_direction vector
    must be normalized before creating this dataclass.
    """

    frustum_corners: np.ndarray
    view_direction: np.ndarray
    near_plane_center: np.ndarray


@dataclass
class TextureConfiguration:
    """Configuration parameters for texture atlas management.

    This dataclass specifies the settings for the fixed-size texture used
    to store selected chunks during rendering. It controls both the texture
    dimensions and optional limits on chunk selection.

    Parameters
    ----------
    texture_width : int
        Edge length of the cubic texture in voxels. The texture is always
        cubic with dimensions (texture_width, texture_width, texture_width).
        This value determines the maximum amount of data that can be stored
        in the texture and should be chosen based on available GPU memory
        and performance requirements. Must be positive.

    Notes
    -----
    The texture_width should be chosen considering:
    - Available GPU memory (larger textures require more memory)
    - Chunk sizes (should accommodate multiple chunks efficiently)
    - Rendering performance (very large textures may impact frame rate)

    """

    texture_width: int


@dataclass
class ChunkSelectionResult:
    """Results from chunk selection algorithms.

    This dataclass encapsulates all outputs from the chunk selection process,
    including which chunks were selected, how the texture is positioned in
    world space, and metadata about the selection.

    Parameters
    ----------
    selected_chunk_mask : np.ndarray
        Boolean array of shape (n_chunks,) indicating which chunks were
        selected for inclusion in the texture. True values correspond to
        chunks that should be rendered, False values to chunks that are
        excluded. The array length matches the total number of chunks in
        the scale level being processed.
    texture_to_world_transform : np.ndarray
        Transformation matrix of shape (4, 4) that converts coordinates from
        texture space to world space. Texture coordinates range from (0,0,0)
        to (texture_width, texture_width, texture_width), while world
        coordinates are in the global coordinate system. This matrix enables
        mapping between the two coordinate systems during rendering.
    texture_bounds_world : tuple[np.ndarray, np.ndarray]
        Texture boundaries in world coordinates as (min_corner, max_corner)
        where each corner is an array of shape (3,) containing (z, y, x)
        coordinates. These bounds define the 3D region of world space that
        is covered by the texture and can be used for spatial queries.
    primary_axis : int
        Primary viewing axis used for texture orientation, encoded as:
        0 = X axis (texture extends primarily along X)
        1 = Y axis (texture extends primarily along Y)
        2 = Z axis (texture extends primarily along Z)
        This axis corresponds to the dimension with the largest component
        in the view direction vector and influences texture positioning.
    n_selected_chunks : int
        Total number of chunks selected for rendering. This is equivalent
        to np.sum(selected_chunk_mask) but provided as a convenience to
        avoid repeated calculations. Useful for logging, debugging, and
        performance monitoring.

    Notes
    -----
    The transformation matrix follows standard 4x4 homogeneous coordinate
    conventions where the bottom row is [0, 0, 0, 1] and the upper-left
    3x3 submatrix contains rotation/scaling while the rightmost column
    contains translation.

    The texture bounds are axis-aligned in world space and define the
    maximum extent of data that could be contained in the texture,
    regardless of whether chunks actually exist in all regions.
    """

    selected_chunk_mask: np.ndarray
    texture_to_world_transform: AffineTransform
    texture_bounds_world: tuple[np.ndarray, np.ndarray]
    primary_axis: int
    n_selected_chunks: int
