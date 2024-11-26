import numpy as np

from cellier.util.chunk import compute_chunk_corners_3d


def test_compute_chunk_corners_3d():
    chunk_corners = compute_chunk_corners_3d(
        array_shape=np.array([10, 10, 10]), chunk_shape=np.array([10, 10, 6])
    )

    expected_corners = np.array(
        [
            [
                [0, 0, 0],
                [0, 0, 6],
                [0, 10, 0],
                [0, 10, 6],
                [10, 0, 0],
                [10, 0, 6],
                [10, 10, 0],
                [10, 10, 6],
            ],
            [
                [0, 0, 6],
                [0, 0, 10],
                [0, 10, 6],
                [0, 10, 10],
                [10, 0, 6],
                [10, 0, 10],
                [10, 10, 6],
                [10, 10, 10],
            ],
        ]
    )
    np.testing.assert_array_equal(chunk_corners, expected_corners)
