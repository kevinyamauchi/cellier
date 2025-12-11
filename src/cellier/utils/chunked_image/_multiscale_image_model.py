"""Data class to represent a multiscale, chunked image."""

from dataclasses import dataclass

import numpy as np
from typing_extensions import Self

from cellier.transform import AffineTransform


@dataclass
class ScaleLevelModel:
    """Represents a single scale/resolution level in a multiscale image.

    Parameters
    ----------
    shape : tuple[int, int, int]
        The dimensions of the entire scale array (z, y, x).
    chunk_shape : tuple[int, int, int]
        The uniform chunk size for this scale (z, y, x).
    transform : AffineTransform
        Transformation from this scale's coordinates to
        scale_0 (full resolution) coordinates.

    Attributes
    ----------
    shape : tuple[int, int, int]
        The dimensions of the entire scale array.
    chunk_shape : tuple[int, int, int]
        The uniform chunk size for this scale.
    transform : AffineTransform
        Transformation to scale_0 coordinates.
    chunk_corners_scale : np.ndarray
        Pre-computed corners for all chunks in scale coordinates.
        Shape is (n_chunks, 8, 3) where n_chunks is the total number of chunks,
        8 represents the corners of each chunk's bounding box,
        and 3 represents (z, y, x) coordinates. These coordinates can
        be used to index the array.
    chunk_corners_scale_0 : np.ndarray
        Pre-computed corners for all chunks in scale_0 (full resolution) coordinates.
        Same shape as chunk_corners_scale.
    chunk_grid_shape : tuple[int, int, int]
        The number of chunks along each dimension.
    _chunk_index_map : dict
        Mapping from chunk grid indices (i, j, k) to linear chunk index.
    """

    shape: tuple[int, int, int]
    chunk_shape: tuple[int, int, int]
    transform: "AffineTransform"

    def __post_init__(self):
        """Initialize computed attributes after dataclass initialization."""
        self.chunk_corners_scale = self._compute_chunk_corners_scale()
        self.chunk_corners_scale_0 = self._compute_chunk_corners_scale_0()
        self._chunk_index_map = self._build_chunk_index_map()

    @property
    def chunk_grid_shape(self) -> tuple[int, int, int]:
        """Get the number of chunks along each dimension.

        Returns
        -------
        tuple[int, int, int]
            Number of chunks in (z, y, x) dimensions.
        """
        return tuple(int(np.ceil(s / c)) for s, c in zip(self.shape, self.chunk_shape))

    @property
    def n_chunks(self) -> int:
        """Get the total number of chunks in this scale level.

        Returns
        -------
        int
            Total number of chunks.
        """
        return int(np.prod(self.chunk_grid_shape))

    def _compute_chunk_corners_scale(self) -> np.ndarray:
        """Compute corners for all chunks in the scale coordinate system.

        Returns
        -------
        np.ndarray
            Array of shape (n_chunks, 8, 3) containing the corners of each
            chunk's bounding box in scale coordinates (can be used to index the array).
        """
        grid_shape = self.chunk_grid_shape
        n_chunks = self.n_chunks
        corners = np.zeros((n_chunks, 8, 3), dtype=np.float32)

        chunk_idx = 0
        for i in range(grid_shape[0]):  # z
            for j in range(grid_shape[1]):  # y
                for k in range(grid_shape[2]):  # x
                    # Calculate chunk bounds
                    z_min = i * self.chunk_shape[0]
                    y_min = j * self.chunk_shape[1]
                    x_min = k * self.chunk_shape[2]

                    z_max = min(z_min + self.chunk_shape[0], self.shape[0])
                    y_max = min(y_min + self.chunk_shape[1], self.shape[1])
                    x_max = min(x_min + self.chunk_shape[2], self.shape[2])

                    # Define 8 corners of the bounding box
                    corners[chunk_idx] = np.array(
                        [
                            [z_min, y_min, x_min],
                            [z_min, y_min, x_max],
                            [z_min, y_max, x_min],
                            [z_min, y_max, x_max],
                            [z_max, y_min, x_min],
                            [z_max, y_min, x_max],
                            [z_max, y_max, x_min],
                            [z_max, y_max, x_max],
                        ],
                        dtype=np.float32,
                    )

                    chunk_idx += 1

        return corners

    def _compute_chunk_corners_scale_0(self) -> np.ndarray:
        """Compute chunk corners in scale_0 (full resolution) coordinates.

        Returns
        -------
        np.ndarray
            Array of shape (n_chunks, 8, 3) containing the corners of each
            chunk's bounding box in scale_0 coordinates.
        """
        # Transform all corners from scale to scale_0 coordinates
        n_chunks, n_corners, n_dims = self.chunk_corners_scale.shape
        corners_flat = self.chunk_corners_scale.reshape(-1, n_dims)
        scale_0_corners_flat = self.transform.map_coordinates(corners_flat)
        return scale_0_corners_flat.reshape(n_chunks, n_corners, n_dims)

    def _build_chunk_index_map(self) -> dict:
        """Build mapping from chunk grid indices to linear chunk index.

        Returns
        -------
        dict
            Mapping from (i, j, k) tuples to linear chunk indices.
        """
        index_map = {}
        grid_shape = self.chunk_grid_shape

        chunk_idx = 0
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                for k in range(grid_shape[2]):
                    index_map[(i, j, k)] = chunk_idx
                    chunk_idx += 1

        return index_map

    def get_chunk_index(self, grid_indices: tuple[int, int, int]) -> int:
        """Get the linear chunk index from grid indices.

        Parameters
        ----------
        grid_indices : tuple[int, int, int]
            Grid indices (i, j, k) of the chunk.

        Returns
        -------
        int
            Linear index of the chunk.

        Raises
        ------
        KeyError
            If the grid indices are out of bounds.
        """
        return self._chunk_index_map[grid_indices]

    def get_chunk_corners(self, chunk_index: int) -> np.ndarray:
        """Get the corners of a specific chunk in scale coordinates.

        Parameters
        ----------
        chunk_index : int
            Linear index of the chunk.

        Returns
        -------
        np.ndarray
            Array of shape (8, 3) containing the chunk's corner coordinates
            in this scale's coordinate system (can be used to index the array).
        """
        return self.chunk_corners_scale[chunk_index].copy()


def compute_scale_transform(downscale_factor: float) -> AffineTransform:
    """Compute affine transformation for a scale level with center-aligned translation.

    This function creates an affine transformation that maps coordinates from a
    downscaled level to the full resolution coordinate system. The transformation
    uses center-aligned translation, meaning each downscaled voxel represents the
    center of a block of voxels in the full resolution.

    For a downscale factor of s, a voxel at position [i, j, k] in the downscaled
    level maps to position
    [i*s + (s-2)/2 + 0.5, j*s + (s-2)/2 + 0.5, k*s + (s-2)/2 + 0.5] in the
    full resolution, which is the center of the (s, s, s) block it represents.

    Parameters
    ----------
    downscale_factor : float
        The isotropic downscaling factor. Must be positive. A value of 2.0 means
        the downscaled image has shape
        (shape[0] // 2, shape[1] // 2,..., shape[n] // 2).

    Returns
    -------
    AffineTransform
        The affine transformation that maps from the downscaled coordinate system
        to the full resolution coordinate system.
    """
    scale = (downscale_factor, downscale_factor, downscale_factor)

    # Translation component for center alignment
    # This ensures that a voxel at [i, j, k] in the downscaled image
    # maps to the center of the corresponding block in full resolution
    offset = (downscale_factor - 2) / 2 + 0.5
    translation = (offset, offset, offset)

    return AffineTransform.from_scale_and_translation(scale, translation)


@dataclass
class MultiscaleImageModel:
    """Container for all scale levels in a multiscale image.

    Parameters
    ----------
    scales : list[ScaleLevelModel]
        List of scale levels ordered from finest to coarsest resolution.

    Attributes
    ----------
    scales : list[ScaleLevelModel]
        List of scale levels ordered from finest to coarsest resolution.
    n_scales : int
        Number of scale levels.
    """

    scales: list[ScaleLevelModel]

    @property
    def n_scales(self) -> int:
        """Get the number of scale levels.

        Returns
        -------
        int
            Number of scale levels.
        """
        return len(self.scales)

    def get_scale(self, scale_index: int) -> ScaleLevelModel:
        """Get a specific scale level.

        Parameters
        ----------
        scale_index : int
            Index of the scale level (0 is finest resolution).

        Returns
        -------
        ScaleLevelModel
            The requested scale level.

        Raises
        ------
        IndexError
            If scale_index is out of bounds.
        """
        return self.scales[scale_index]

    def get_full_extent_at_scale(
        self, scale_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the bounding box of the entire dataset at a specific scale.

        Parameters
        ----------
        scale_index : int
            Index of the scale level.

        Returns
        -------
        min_corner : np.ndarray
            Minimum corner of the bounding box in scale coordinates (z, y, x).
        max_corner : np.ndarray
            Maximum corner of the bounding box in scale coordinates (z, y, x).
        """
        scale = self.scales[scale_index]
        min_corner = np.array([0, 0, 0], dtype=np.float32)
        max_corner = np.array(scale.shape, dtype=np.float32)

        return min_corner, max_corner

    def get_full_extent_world(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the bounding box of the entire dataset in world coordinates.

        Returns
        -------
        min_corner : np.ndarray
            Minimum corner of the bounding box in world coordinates (z, y, x).
        max_corner : np.ndarray
            Maximum corner of the bounding box in world coordinates (z, y, x).
        """
        # Use the finest scale (index 0) to compute world extent
        scale = self.scales[0]
        min_corner = np.squeeze(scale.transform.map_coordinates(np.array([0, 0, 0])))
        max_corner = np.squeeze(scale.transform.map_coordinates(np.array(scale.shape)))

        return min_corner, max_corner

    @classmethod
    def from_shape_and_scales(
        cls,
        shape: tuple[int, int, int],
        chunk_shapes: list[tuple[int, int, int]],
        downscale_factors: list[float],
    ) -> Self:
        """Create a multiscale image model from the image shape and scales.

        Parameters
        ----------
        shape : tuple[int, int, int]
            The full resolution shape of the image (z, y, x).
        chunk_shapes : list[tuple[int, int, int]]
            List of chunk shapes for each scale level in (z, y, x).
        downscale_factors : list[float]
            List of downscale factors for each scale level. A downscale factor of 2.0
            means the scale level has shape
            (shape[0] // 2, shape[1] // 2,..., shape[n] // 2).
            Must be isotropic (same factor for all dimensions).

        Returns
        -------
        MultiscaleImageModel
            The constructed multiscale image model containing all scales.
        """
        if len(chunk_shapes) != len(downscale_factors):
            raise ValueError(
                "chunk_shapes and downscale_factors must have the same length"
            )

        if downscale_factors[0] != 1.0:
            raise ValueError(
                f"First downscale factor must be 1.0 for full resolution, "
                f"got {downscale_factors[0]}"
            )

        scales = []

        for chunk_shape, factor in zip(chunk_shapes, downscale_factors):
            # Compute shape at this scale level
            scale_shape = tuple(int(np.ceil(s / factor)) for s in shape)

            # Create transformation for this scale
            transform = compute_scale_transform(factor)

            # Create the scale level
            scale_level = ScaleLevelModel(
                shape=scale_shape, chunk_shape=chunk_shape, transform=transform
            )

            scales.append(scale_level)

        return MultiscaleImageModel(scales=scales)
