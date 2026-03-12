"""Frustum culling for chunked volumetric images."""

import numpy as np

from cellier.transform import AffineTransform
from cellier.utils.chunked_image._multiscale_image_model import ScaleLevelModel
from cellier.utils.geometry import frustum_planes_from_corners, points_in_frustum


class ChunkCuller:
    """Determine which chunks of a scale level are visible in a view frustum.

    Culling is performed in scale_0 (full-resolution array) coordinates using
    pre-computed chunk corners stored on ``ScaleLevelModel``. The view frustum
    (supplied in world coordinates) is inverse-mapped into scale_0 space before
    any plane tests are applied.

    A chunk is considered **visible** if *any* of its 8 corner points lies
    inside the frustum. This conservative criterion prevents false culling of
    chunks that straddle a frustum boundary.
    """

    def cull_chunks(
        self,
        scale_level: ScaleLevelModel,
        frustum_corners: np.ndarray,
        world_transform: AffineTransform,
    ) -> np.ndarray:
        """Return a boolean visibility mask for every chunk in the scale level.

        Parameters
        ----------
        scale_level : ScaleLevelModel
            Scale level whose chunks are to be tested. Uses the pre-computed
            ``chunk_corners_scale_0`` attribute (shape ``(n_chunks, 8, 3)``).
        frustum_corners : np.ndarray
            View frustum corners in **world** coordinates. Shape ``(2, 4, 3)``.
            First axis: near (index 0) and far (index 1) planes.
            Second axis: corners in order
            ``(left_bottom, right_bottom, right_top, left_top)``.
            Third axis: ``(z, y, x)`` coordinates.
        world_transform : AffineTransform
            Transform whose *forward* direction maps scale_0 → world and whose
            *inverse* maps world → scale_0.
            ``MultiscaleImageModel.world_transform`` should be passed here.

        Returns
        -------
        np.ndarray
            Boolean array of shape ``(n_chunks,)`` where ``True`` means the
            chunk has at least one corner inside the frustum.
        """
        # Step 1: transform frustum corners from world → scale_0
        corners_flat = frustum_corners.reshape(-1, 3)  # (8, 3)
        corners_scale0_flat = world_transform.imap_coordinates(corners_flat)
        frustum_corners_scale0 = corners_scale0_flat.reshape(2, 4, 3)

        # Step 2: compute inward-pointing planes of the frustum in scale_0 space
        planes = frustum_planes_from_corners(frustum_corners_scale0)  # (6, 4)

        # Step 3: flatten all chunk corners to a single point array
        n_chunks = scale_level.chunk_corners_scale_0.shape[0]
        all_corners_flat = scale_level.chunk_corners_scale_0.reshape(
            n_chunks * 8, 3
        )  # (n_chunks*8, 3)

        # Step 4: test all corners against all planes in one vectorised call
        point_mask_flat = points_in_frustum(
            points=all_corners_flat, planes=planes
        )  # (n_chunks*8,)

        # Step 5: reduce to per-chunk visibility — True if any corner is inside
        point_mask = point_mask_flat.reshape(n_chunks, 8)  # (n_chunks, 8)
        chunk_mask = np.any(point_mask, axis=1)  # (n_chunks,)

        return chunk_mask
