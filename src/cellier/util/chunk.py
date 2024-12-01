"""Utilities for dealing with array chunks."""

import numpy as np

from cellier.util.geometry import (
    frustum_planes_from_corners,
    near_far_plane_edge_lengths,
    points_in_frustum,
)


def compute_chunk_corners_3d(
    array_shape: np.ndarray, chunk_shape: np.ndarray
) -> np.ndarray:
    """Compute the corners for each chunk of a 3D array.

    Parameters
    ----------
    array_shape : np.ndarray
        The shape of the array.
    chunk_shape : np.ndarray
        The shape of the chunks.
        Axes must be aligned with the array.

    Returns
    -------
    np.ndarray
        (N, 8, 3) array containing the coordinates for the corners
        of each chunk. Coordinates are the array indices.
    """
    # determine the number of chunks along each dimension
    n_chunks = np.ceil(array_shape / chunk_shape)

    # Iterate over chunks in the 3D grid
    all_corners = []
    for chunk_index_0 in range(int(n_chunks[0])):
        for chunk_index_1 in range(int(n_chunks[1])):
            for chunk_index_2 in range(int(n_chunks[2])):
                # Calculate start and stop indices for this chunk
                min_0 = chunk_index_0 * chunk_shape[0]
                min_1 = chunk_index_1 * chunk_shape[1]
                min_2 = chunk_index_2 * chunk_shape[2]

                max_0 = min(min_0 + chunk_shape[0], array_shape[0])
                max_1 = min(min_1 + chunk_shape[1], array_shape[1])
                max_2 = min(min_2 + chunk_shape[2], array_shape[2])

                # Define the 8 corners of the chunk
                corners = np.array(
                    [
                        [min_0, min_1, min_2],
                        [min_0, min_1, max_2],
                        [min_0, max_1, min_2],
                        [min_0, max_1, max_2],
                        [max_0, min_1, min_2],
                        [max_0, min_1, max_2],
                        [max_0, max_1, min_2],
                        [max_0, max_1, max_2],
                    ]
                )
                all_corners.append(corners)

    return np.array(all_corners, dtype=int)


class ChunkedArray3D:
    """Data structure for querying chunks from a chunked array.

    Transforms are defined from the chunked array to the parent coordinate system.
    """

    def __init__(
        self,
        array_shape,
        chunk_shape,
        scale: tuple[float, float, float] = (1, 1, 1),
        translation: tuple[float, float, float] = (0, 0, 0),
    ):
        self.array_shape = np.asarray(array_shape)
        self.chunk_shape = np.asarray(chunk_shape)
        self.scale = np.asarray(scale)
        self.translation = np.asarray(translation)
        self._chunk_coordinates = compute_chunk_corners_3d(
            array_shape=self.array_shape, chunk_shape=self.chunk_shape
        )

    @property
    def n_chunks(self) -> int:
        """Number of chunks in the array."""
        return self.chunk_corners.shape[0]

    @property
    def chunk_corners(self) -> np.ndarray:
        """(N, 8, 3) array containing the coordinates for the corners of each chunk.

        Coordinates are ordered:
            [min_0, min_1, min_2],
            [min_0, min_1, max_2],
            [min_0, max_1, min_2],
            [min_0, max_1, max_2],
            [max_0, min_1, min_2],
            [max_0, min_1, max_2],
            [max_0, max_1, min_2],
            [max_0, max_1, max_2],
        """
        return self._chunk_coordinates

    @property
    def chunk_corners_flat(self) -> np.ndarray:
        """(N*8, 3) array containing the corners of all chunks."""
        n_corners = self.chunk_corners.shape[0] * 8
        return self.chunk_corners.reshape(n_corners, 3)

    @property
    def chunk_centers(self) -> np.ndarray:
        """(N, 3) array containing the center coordinate of each chunk."""
        return np.mean(self.chunk_corners, axis=1)

    def chunks_in_frustum(self, planes: np.ndarray, mode="any") -> np.ndarray:
        """Determine which chunks are in the frustum plane."""
        points_mask = points_in_frustum(
            points=self.chunk_corners_flat, planes=planes
        ).reshape(self.n_chunks, 8)

        if mode == "any":
            return np.any(points_mask, axis=1)
        elif mode == "all":
            return np.all(points_mask, axis=1)
        else:
            raise ValueError(f"{mode} is not a valid. Should be any or all")


class MultiScaleChunkedArray3D:
    """A data model for a multiscale chunked array."""

    def __init__(self, scales: list[ChunkedArray3D]):
        self._scales = scales

        self._min_voxel_size_local = np.array(
            [np.min(scale_level.scale) for scale_level in self.scales]
        )

        self._n_scales = len(self.scales)

    @property
    def scales(self) -> list[ChunkedArray3D]:
        """List of ChunkedArray3D."""
        return self._scales

    @property
    def n_scales(self) -> int:
        """The number of scale levels."""
        return self._n_scales

    @property
    def min_voxel_size_local(self) -> np.ndarray:
        """The minimum edge length for a voxel at each scale.

        Size is in the local coordinate system.
        """
        return self._min_voxel_size_local

    def _select_scale_by_logical_voxel_size(
        self,
        frustum_width_local: float,
        frustum_height_local: float,
        width_logical: int,
        height_logical: int,
    ) -> ChunkedArray3D:
        """Select the scale based on the size of the logical voxel.

        This method tries to select a scale where the size of the voxel
        is closest to the one logical pixel.
        """
        # get the smallest size of the logical pixels
        logical_pixel_width_local = frustum_width_local / width_logical
        logical_pixel_height_local = frustum_height_local / height_logical
        logical_pixel_local = min(logical_pixel_width_local, logical_pixel_height_local)

        pixel_size_difference = self.min_voxel_size_local - logical_pixel_local

        for level_index in reversed(range(self.n_scales)):
            if pixel_size_difference[level_index] <= 0:
                selected_level_index = min(self.n_scales - 1, level_index + 1)
                return self.scales[selected_level_index]

        # if none work, return the highest resolution
        return self.scales[0]

    def _select_scale_by_frustum_width(
        self,
        frustum_corners: np.ndarray,
        texture_shape: np.ndarray,
        width_factor: float,
    ) -> ChunkedArray3D:
        # get the characteristic width of the frustum
        frustum_width = np.max(near_far_plane_edge_lengths(corners=frustum_corners))

        for chunked_array in self.scales:
            texture_width = np.min(chunked_array.scale * texture_shape)

            if texture_width >= (frustum_width * width_factor):
                return chunked_array
        # if none meet the criteria, return the lowest resolution
        return self.scales[-1]

    def _select_scale_by_texture_bounding_box(
        self,
        frustum_corners: np.ndarray,
        texture_shape: np.ndarray,
        width_factor: float,
    ) -> ChunkedArray3D:
        for level_index in reversed(range(self.n_scales)):
            chunked_array = self.scales[level_index]
            flat_corners = frustum_corners.reshape(8, 3)
            transformed_flat_corners = (
                flat_corners / chunked_array.scale
            ) - chunked_array.translation
            transformed_corners = transformed_flat_corners.reshape(2, 4, 3)

            frustum_planes = frustum_planes_from_corners(transformed_corners)
            chunk_mask = chunked_array.chunks_in_frustum(
                planes=frustum_planes, mode="any"
            )
            chunks_to_update = chunked_array.chunk_corners[chunk_mask]
            n_chunks_to_update = chunks_to_update.shape[0]
            chunks_to_update_flat = chunks_to_update.reshape(
                (n_chunks_to_update * 8, 3)
            )
            min_corner_all = np.min(chunks_to_update_flat, axis=0)
            max_corner_all = np.max(chunks_to_update_flat, axis=0)

            required_shape = max_corner_all - min_corner_all
            if np.any(required_shape > texture_shape):
                # if this resolution would require too many chunks,
                # take one resolution lower (scales go from highest to lowest res)
                selected_level = min(self.n_scales, level_index + 1)
                return self.scales[selected_level]
        # if all scales fit, use the highest res one
        # (scales go from highest to lowest res)
        return self.scales[0]

    def scale_from_frustum(
        self,
        frustum_corners: np.ndarray,
        width_logical: int,
        height_logical: int,
        texture_shape: np.ndarray,
        width_factor: float,
        method: str = "width",
    ) -> ChunkedArray3D:
        """Determine the appropriate scale from the frustum corners."""
        if method == "width":
            return self._select_scale_by_frustum_width(
                frustum_corners=frustum_corners,
                texture_shape=texture_shape,
                width_factor=width_factor,
            )
        elif method == "full_texture_size":
            return self._select_scale_by_texture_bounding_box(
                frustum_corners=frustum_corners,
                texture_shape=texture_shape,
                width_factor=width_factor,
            )
        elif method == "logical_pixel_size":
            near_plane = frustum_corners[0]
            width_local = np.linalg.norm(near_plane[1, :] - near_plane[0, :])
            height_local = np.linalg.norm(near_plane[3, :] - near_plane[0, :])
            return self._select_scale_by_logical_voxel_size(
                frustum_width_local=width_local,
                frustum_height_local=height_local,
                width_logical=width_logical,
                height_logical=height_logical,
            )
        else:
            raise ValueError(f"Unknown method {method}.")
