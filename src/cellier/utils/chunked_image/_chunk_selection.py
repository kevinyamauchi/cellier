"""Axis-aligned grid-snapped texture positioning strategy.

This module implements the core positioning algorithm that places a cubic texture
in world space by centering it on the camera near plane and snapping to chunk
grid boundaries.
"""

import numpy as np

from cellier.transform import AffineTransform
from cellier.utils.chunked_image._base import (
    ChunkFilteringStrategy,
    TexturePositioningStrategy,
)
from cellier.utils.chunked_image._data_classes import (
    ChunkSelectionResult,
    TextureConfiguration,
    ViewParameters,
)
from cellier.utils.chunked_image._multiscale_image_model import ScaleLevelModel


class AxisAlignedTexturePositioning(TexturePositioningStrategy):
    """Position texture centered on the camera near plane.

    The texture is placed as a fixed-size axis-aligned cube whose center is
    the camera near-plane centre (transformed to scale_N space), snapped to
    the chunk grid for cache stability.  The window is clamped so it stays
    within the data extent.
    """

    def position_texture(
        self,
        view_params: ViewParameters,
        scale_level: ScaleLevelModel,
        texture_config: TextureConfiguration,
    ) -> tuple[AffineTransform, tuple[np.ndarray, np.ndarray], int]:
        """Position texture centered on the camera near plane.

        Parameters
        ----------
        view_params : ViewParameters
            Camera view information.  ``near_plane_center`` is used to anchor
            the texture; ``view_direction`` determines the primary axis.
        scale_level : ScaleLevelModel
            Scale level metadata used for coordinate transforms and clamping.
        texture_config : TextureConfiguration
            Texture configuration settings including ``texture_width``.

        Returns
        -------
        texture_to_scale_transform : AffineTransform
            Pure-translation transform mapping texture-space origin to the
            positioning corner in scale_N coordinates.
        texture_bounds : tuple[np.ndarray, np.ndarray]
            ``(texture_min, texture_max)`` in scale_N coordinates.
        primary_axis : int
            Index of the dominant viewing axis (0, 1, or 2).
        """
        # Step 1: Transform near_plane_center (a point, w=1) to scale_N
        near_scale = scale_level.transform.imap_coordinates(
            view_params.near_plane_center.reshape(1, -1)
        ).flatten()

        # Step 2: Compute chunk-grid-snapped positioning corner
        positioning_corner = self._compute_positioning_corner(
            near_scale, scale_level, texture_config
        )

        # Step 3: Texture bounds in scale_N space
        texture_bounds = self._calculate_texture_bounds(
            positioning_corner, texture_config
        )

        # Step 4: Texture-to-scale transform (pure translation)
        texture_to_scale_transform = AffineTransform.from_translation(
            positioning_corner
        )

        # Step 5: Primary axis from world-space view direction
        primary_axis = self._determine_primary_axis(view_params.view_direction)

        return texture_to_scale_transform, texture_bounds, primary_axis

    def _compute_positioning_corner(
        self,
        near_center_scale: np.ndarray,
        scale_level: ScaleLevelModel,
        texture_config: TextureConfiguration,
    ) -> np.ndarray:
        """Compute the texture's min corner in scale_N space.

        Centers the texture on ``near_center_scale``, snaps to the chunk grid,
        then clamps so the window stays within ``[0, shape]``.

        Parameters
        ----------
        near_center_scale : np.ndarray
            Camera near-plane centre in scale_N coordinates, shape (3,).
        scale_level : ScaleLevelModel
            Provides ``chunk_shape`` and ``shape`` for snapping and clamping.
        texture_config : TextureConfiguration
            Provides ``texture_width``.

        Returns
        -------
        np.ndarray
            Positioning corner (min corner of the texture window), shape (3,).
        """
        half = texture_config.texture_width / 2.0
        raw_corner = near_center_scale - half

        # Snap to chunk grid for cache stability
        chunk = np.array(scale_level.chunk_shape, dtype=float)
        snapped = np.floor(raw_corner / chunk) * chunk

        # Clamp so the full texture window fits within the data extent
        max_valid = np.maximum(
            0.0,
            np.array(scale_level.shape, dtype=float) - texture_config.texture_width,
        )
        return np.clip(snapped, 0.0, max_valid)

    def _calculate_texture_bounds(
        self, positioning_corner: np.ndarray, texture_config: TextureConfiguration
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (texture_min, texture_max) in scale_N coordinates."""
        texture_min = positioning_corner.copy()
        texture_max = positioning_corner + texture_config.texture_width
        return texture_min, texture_max

    def _determine_primary_axis(self, view_direction: np.ndarray) -> int:
        """Return index of the axis with the largest absolute view component."""
        return int(np.argmax(np.abs(view_direction)))


class TextureBoundsFiltering(ChunkFilteringStrategy):
    """Filters chunks based on complete inclusion within texture bounds.

    This strategy implements spatial filtering that selects chunks where all
    8 corner points fall within the texture bounds. This ensures that selected
    chunks can be completely contained within the texture.
    """

    def filter_chunks(
        self,
        chunk_indices: np.ndarray,
        chunk_corners_world: np.ndarray,
        texture_bounds_world: tuple[np.ndarray, np.ndarray],
        view_params: "ViewParameters",
        texture_config: "TextureConfiguration",
    ) -> np.ndarray:
        """Filter chunks based on complete inclusion within texture bounds.

        Parameters
        ----------
        chunk_indices : np.ndarray
            Array of shape (n_candidate_chunks,) containing linear indices
            for candidate chunks.
        chunk_corners_world : np.ndarray
            Array of shape (n_candidate_chunks, 8, 3) containing the corner
            coordinates of each candidate chunk in scale/world space.
        texture_bounds_world : tuple[np.ndarray, np.ndarray]
            Texture bounds as (min_corner, max_corner), each (3,).
        view_params : ViewParameters
            Unused; kept for interface consistency.
        texture_config : TextureConfiguration
            Unused; kept for interface consistency.

        Returns
        -------
        np.ndarray
            Boolean array of shape (n_candidate_chunks,) where True means all
            8 corners of the chunk lie within the texture bounds.
        """
        if chunk_indices.size == 0:
            return np.array([], dtype=bool)

        texture_min, texture_max = texture_bounds_world
        return self._chunks_within_bounds_vectorized(
            chunk_corners_world, texture_min, texture_max
        )

    def _chunks_within_bounds_vectorized(
        self,
        chunk_corners_world: np.ndarray,
        texture_min: np.ndarray,
        texture_max: np.ndarray,
    ) -> np.ndarray:
        within_min = chunk_corners_world >= texture_min[None, None, :]
        within_max = chunk_corners_world <= texture_max[None, None, :]
        return np.all(within_min & within_max, axis=(1, 2))


class ChunkSelector:
    """Coordinates texture positioning and chunk selection.

    Pipeline
    --------
    1. Position the texture centered on the camera near plane (scale_N space).
    2. Find all chunks whose corners lie completely within the texture bounds.
    3. Intersect with the frustum-visible mask (if provided).
    4. Return results transformed back to world coordinates.
    """

    def __init__(
        self,
        positioning_strategy: TexturePositioningStrategy,
        filtering_strategy: ChunkFilteringStrategy,
    ):
        self._positioning_strategy = positioning_strategy
        self._filtering_strategy = filtering_strategy

    def select_chunks(
        self,
        scale_level: ScaleLevelModel,
        view_params: ViewParameters,
        texture_config: TextureConfiguration,
        frustum_visible_chunks: np.ndarray | None = None,
    ) -> ChunkSelectionResult:
        """Select chunks for rendering in a fixed-size texture atlas.

        Parameters
        ----------
        scale_level : ScaleLevelModel
            Scale level with chunk metadata and coordinate transforms.
        view_params : ViewParameters
            Camera state in world coordinates.
        texture_config : TextureConfiguration
            Texture size constraints.
        frustum_visible_chunks : np.ndarray, optional
            Boolean mask (n_chunks,) from frustum culling.  If None all chunks
            are treated as frustum-visible.

        Returns
        -------
        ChunkSelectionResult
        """
        # Step 1: Transform view params to scale_N for positioning
        view_params_scale = self._transform_view_params_to_scale(
            view_params, scale_level
        )

        # Step 2: Position texture from near_plane_center (scale_N)
        transform_scale, texture_bounds_scale, primary_axis = (
            self._positioning_strategy.position_texture(
                view_params_scale,
                scale_level,
                texture_config,
            )
        )

        # Step 3: Find ALL chunks inside the texture bounds
        all_indices = np.arange(scale_level.n_chunks)
        texture_mask = self._filtering_strategy.filter_chunks(
            all_indices,
            scale_level.chunk_corners_scale,
            texture_bounds_scale,
            view_params_scale,
            texture_config,
        )

        # Step 4: Intersect with frustum mask
        if frustum_visible_chunks is not None:
            full_scale_mask = texture_mask & frustum_visible_chunks
        else:
            full_scale_mask = texture_mask

        n_selected = int(np.sum(full_scale_mask))

        if n_selected == 0:
            identity_transform = AffineTransform.from_translation(np.zeros(3))
            zero_bounds = (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
            return ChunkSelectionResult(
                selected_chunk_mask=full_scale_mask,
                texture_to_world_transform=identity_transform,
                texture_bounds_world=zero_bounds,
                primary_axis=primary_axis,
                n_selected_chunks=0,
            )

        # Step 5: Compose transforms and convert bounds back to world coordinates
        transform_world = self._compose_transforms(
            scale_level.transform, transform_scale
        )
        texture_bounds_world = self._transform_bounds_to_world(
            texture_bounds_scale, scale_level.transform
        )

        return ChunkSelectionResult(
            selected_chunk_mask=full_scale_mask,
            texture_to_world_transform=transform_world,
            texture_bounds_world=texture_bounds_world,
            primary_axis=primary_axis,
            n_selected_chunks=n_selected,
        )

    def _transform_view_params_to_scale(
        self, view_params: ViewParameters, scale_level: ScaleLevelModel
    ) -> ViewParameters:
        """Transform view parameters from world to scale_N coordinates."""
        # Frustum corners are points (w=1)
        frustum_flat = view_params.frustum_corners.reshape(-1, 3)
        frustum_scale = scale_level.transform.imap_coordinates(frustum_flat)
        frustum_corners_scale = frustum_scale.reshape(view_params.frustum_corners.shape)

        # View direction is a vector (w=0) — translation must not be applied
        view_direction_scale = scale_level.transform.imap_direction(
            view_params.view_direction.reshape(1, -1)
        ).flatten()

        # Near plane center is a point (w=1)
        near_scale = scale_level.transform.imap_coordinates(
            view_params.near_plane_center.reshape(1, -1)
        ).flatten()

        return ViewParameters(
            frustum_corners=frustum_corners_scale.astype(np.float32),
            view_direction=view_direction_scale.astype(np.float32),
            near_plane_center=near_scale.astype(np.float32),
        )

    def _compose_transforms(
        self, scale_to_world_transform, texture_to_scale_transform
    ) -> AffineTransform:
        """Compose texture→scale→world into a single transform."""
        new_matrix = scale_to_world_transform.matrix @ texture_to_scale_transform.matrix
        return AffineTransform(matrix=new_matrix)

    def _transform_bounds_to_world(
        self,
        texture_bounds_scale: tuple[np.ndarray, np.ndarray],
        scale_to_world_transform,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform texture bounds from scale_N to world coordinates."""
        min_s, max_s = texture_bounds_scale
        corners_world = scale_to_world_transform.map_coordinates(
            np.vstack([min_s, max_s])
        )
        return corners_world[0].astype(np.float32), corners_world[1].astype(np.float32)
