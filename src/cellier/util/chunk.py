"""Utilities for dealing with array chunks."""

import numpy as np


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


class ChunkData3D:
    """Data structure for querying chunks from a chunked array."""

    def __init__(self, array_shape, chunk_shape):
        self.array_shape = np.asarray(array_shape)
        self.chunk_shape = np.asarray(chunk_shape)
        self._chunk_coordinates = compute_chunk_corners_3d(
            array_shape=self.array_shape, chunk_shape=self.chunk_shape
        )

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
    def chunk_centers(self) -> np.ndarray:
        """(N, 3) array containing the center coordinate of each chunk."""
        return np.mean(self.chunk_corners, axis=1)
