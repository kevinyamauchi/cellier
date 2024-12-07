"""Utilities for dealing with array chunks."""

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

from cellier.models.data_stores.base_data_store import DataStoreSlice
from cellier.util.geometry import (
    frustum_planes_from_corners,
    near_far_plane_edge_lengths,
    points_in_frustum,
)


def compute_chunk_corners_3d(
    array_shape: np.ndarray, chunk_shape: np.ndarray
) -> np.ndarray:
    """Compute the corners for each chunk of a 3D array.

    todo: jit?

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


def generate_chunk_requests_from_frustum(
    frustum_corners: np.ndarray,
    chunk_corners: np.ndarray,
    scale_index: int,
    scene_id: str,
    visual_id: str,
    mode: str = "any",
) -> tuple[list["ImageDataStoreChunk"], tuple[int, int, int], tuple[int, int, int]]:
    """Generate a request for each chunk that is inside of a view frustum.

    Parameters
    ----------
    frustum_corners : np.ndarray
        The corners of the view frustum in the coordinates of the scale.
        That is they should be scaled and translated to match the coordinates
        of the array being passed in.
        The corner coordinates should be in an array shaped (2, 4, 3).
        The first axis corresponds to the frustum plane (near, far).
        The second corresponds to the corner within the plane
        ((left, bottom), (right, bottom), (right, top), (left, top))
        The third corresponds to the coordinates of that corner.
    chunk_corners : np.ndarray
        (n, 8, 3) array containing the corners of each chunk.
    scale_index : int
        The index for the scale level these chunks come from.
    scene_id : str
        The unique identifier for the Scene.
    visual_id : str
        The unique identifier for the visual the chunks belong to.
    mode : str
        The mode for determining which chunks are in the frustum.
        "any" takes chunks where any of the corners are inside of the frustum.
        "all" takes chunks where all of the corners are inside the frustum.
        Default value is "any".


    """
    # get the planes of the frustum
    planes = frustum_planes_from_corners(frustum_corners)

    # determine which chunks are in the frustum
    n_chunks = chunk_corners.shape[0]
    chunk_corners_flat = chunk_corners.reshape((n_chunks * 8, -1))
    points_mask = points_in_frustum(points=chunk_corners_flat, planes=planes).reshape(
        n_chunks, 8
    )

    if mode == "any":
        chunk_mask = np.any(points_mask, axis=1)
    elif mode == "all":
        chunk_mask = np.all(points_mask, axis=1)
    else:
        raise ValueError(f"{mode} is not a valid. Should be any or all")

    chunks_to_request = chunk_corners[chunk_mask]

    if len(chunks_to_request) == 0:
        return [], (), ()

    # get the lower-left bounds of the array
    n_chunks_to_request = chunks_to_request.shape[0]
    chunks_to_request_flat = chunks_to_request.reshape((n_chunks_to_request * 8, -1))
    min_corner_all_local = np.min(chunks_to_request_flat, axis=0)
    max_corner_all_local = np.max(chunks_to_request_flat, axis=0)
    new_texture_shape = max_corner_all_local - min_corner_all_local

    # make a request for each chunk
    chunk_requests = []
    for chunk in chunks_to_request:
        # get the corners of the chunk in the array index coordinates
        min_corner_array = chunk[0]
        max_corner_array = chunk[7]

        # get the corners of the chunk in the texture index coordinates
        min_corner_texture = min_corner_array - min_corner_all_local
        max_corner_texture = max_corner_array - min_corner_all_local

        if np.any(max_corner_texture > new_texture_shape) or np.any(
            min_corner_texture > new_texture_shape
        ):
            print(f"skipping: {min_corner_array}")
            continue

        chunk_requests.append(
            ImageDataStoreChunk(
                resolution_level=scale_index,
                array_coordinate_start=min_corner_array[[2, 1, 0]],
                array_coordinate_end=max_corner_array[[2, 1, 0]],
                # array_coordinate_start=min_corner_array,
                # array_coordinate_end=max_corner_array,
                texture_coordinate_start=min_corner_texture,
                scene_id=scene_id,
                visual_id=visual_id,
            )
        )

    return chunk_requests, new_texture_shape, min_corner_all_local


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


@dataclass(frozen=True)
class ImageDataStoreChunk(DataStoreSlice):
    """Class containing data to access a chunk in an image data store.

    Parameters
    ----------
    scene_id : str
        The UID of the scene the visual is rendered in.
    visual_id : str
        The UID of the corresponding visual.
    resolution_level : int
        The resolution level to render where 0 is the highest resolution
        and high numbers correspond with more down sampling.
    texture_coordinate_start : Union[Tuple[int, int], Tuple[int, int, int]]
        The min coordinates of the chunk in the texture it will be rendered in.
    array_coordinate_start : Tuple[int, ...]
        The min coordinates of the chunk in the array coordinates.
    array_coordinate_max : Tuple[int, ...]
        The max coordinates of the chunk in the array coordinates.
    """

    texture_coordinate_start: Union[Tuple[int, int], Tuple[int, int, int]]
    array_coordinate_start: Tuple[int, ...]
    array_coordinate_end: Tuple[int, ...]
