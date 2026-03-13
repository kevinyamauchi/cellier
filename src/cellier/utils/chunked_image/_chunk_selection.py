"""Axis-aligned grid-snapped texture positioning strategy.

This module implements the core positioning algorithm that places a cubic texture
in world space by centering it on the camera position projected onto the data and
snapping to chunk grid boundaries.
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
    """Position texture centered on the point where the camera ray meets the data.

    The texture is placed as a fixed-size axis-aligned cube whose centre is
    the projection of the data-volume centre onto the camera ray (in scale_N
    space), snapped to the chunk grid for cache stability.  The window is
    clamped so it stays within the data extent.

    For orthographic frustums (or degenerate cameras) the anchor falls back
    to the near-plane centre.
    """

    def position_texture(
        self,
        view_params: ViewParameters,
        scale_level: ScaleLevelModel,
        texture_config: TextureConfiguration,
    ) -> tuple[AffineTransform, tuple[np.ndarray, np.ndarray], int]:
        """Position texture centred on the camera-ray / data-centre intersection.

        Parameters
        ----------
        view_params : ViewParameters
            Camera view information **already transformed to scale_N space**.
            ``frustum_corners``, ``view_direction``, and ``near_plane_center``
            are all in scale_N coordinates.
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
        # Compute chunk-grid-snapped positioning corner
        positioning_corner = self._compute_positioning_corner(
            view_params, scale_level, texture_config
        )

        # Texture bounds in scale_N space
        texture_bounds = self._calculate_texture_bounds(
            positioning_corner, texture_config
        )

        # Texture-to-scale transform (pure translation)
        texture_to_scale_transform = AffineTransform.from_translation(
            positioning_corner
        )

        # Primary axis from scale_N view direction
        primary_axis = self._determine_primary_axis(view_params.view_direction)

        return texture_to_scale_transform, texture_bounds, primary_axis

    def _recover_camera_position_scale(
        self,
        view_params_scale: ViewParameters,
    ) -> np.ndarray | None:
        """Recover camera position in scale_N from the frustum corners.

        Uses the ratio of near-to-far plane heights to locate the perspective
        apex.  Returns *None* for orthographic frustums (far_h ≈ near_h) or
        degenerate inputs.

        Parameters
        ----------
        view_params_scale : ViewParameters
            View parameters already in scale_N space.

        Returns
        -------
        np.ndarray or None
            Camera position in scale_N coordinates, shape (3,), or None.
        """
        near_corners = view_params_scale.frustum_corners[0]  # (4, 3)
        far_corners = view_params_scale.frustum_corners[1]  # (4, 3)

        near_h = float(np.linalg.norm(near_corners[3] - near_corners[0]))
        far_h = float(np.linalg.norm(far_corners[3] - far_corners[0]))

        if far_h <= near_h * 1.001:
            return None  # orthographic or near-degenerate

        view_dir = view_params_scale.view_direction.astype(float)
        view_dir_norm = float(np.linalg.norm(view_dir))
        if view_dir_norm < 1e-9:
            return None
        view_dir_unit = view_dir / view_dir_norm

        near_center = near_corners.mean(axis=0).astype(float)
        far_center = far_corners.mean(axis=0).astype(float)

        d_near_far = float(np.dot(far_center - near_center, view_dir_unit))
        if d_near_far <= 0:
            return None

        k = far_h / near_h  # > 1 for perspective
        near_dist = d_near_far / (k - 1.0)
        return near_center - near_dist * view_dir_unit

    def _compute_positioning_corner(
        self,
        view_params_scale: ViewParameters,
        scale_level: ScaleLevelModel,
        texture_config: TextureConfiguration,
    ) -> np.ndarray:
        """Compute the texture's min corner in scale_N space.

        Projects the data-volume centre onto the camera ray to obtain an anchor
        that is always inside the data, then centres the texture window on that
        anchor.  Snaps to the chunk grid for cache stability and clamps so the
        window stays within ``[0, shape]``.

        For orthographic frustums (or degenerate cameras), falls back to
        centering on ``near_plane_center``.

        Parameters
        ----------
        view_params_scale : ViewParameters
            Camera view information already in scale_N coordinates.
        scale_level : ScaleLevelModel
            Provides ``chunk_shape`` and ``shape`` for snapping and clamping.
        texture_config : TextureConfiguration
            Provides ``texture_width``.

        Returns
        -------
        np.ndarray
            Positioning corner (min corner of the texture window), shape (3,).
        """
        data_center_N = np.array(scale_level.shape, dtype=float) / 2.0

        cam_pos_N = self._recover_camera_position_scale(view_params_scale)

        if cam_pos_N is None:
            # Orthographic fallback: centre on near_plane_center
            anchor_N = view_params_scale.near_plane_center.copy().astype(float)
        else:
            view_dir = view_params_scale.view_direction.astype(float)
            view_dir_norm = float(np.linalg.norm(view_dir))
            if view_dir_norm < 1e-9:
                anchor_N = data_center_N.copy()
            else:
                view_dir_unit = view_dir / view_dir_norm
                # Project data centre onto camera ray to get the anchor depth
                t = float(np.dot(data_center_N - cam_pos_N, view_dir_unit))
                t = max(t, 0.0)  # don't project behind the camera
                anchor_N = cam_pos_N + t * view_dir_unit
                # Clamp anchor to data extent before centering
                anchor_N = np.clip(
                    anchor_N, 0.0, np.array(scale_level.shape, dtype=float)
                )

        half = texture_config.texture_width / 2.0
        raw_corner = anchor_N - half

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
    1. Transform view parameters from world → scale_N (via world_transform and
       scale_level.transform).
    2. Position the texture centred on the camera-ray / data-centre intersection.
    3. Find all chunks whose corners lie completely within the texture bounds.
    4. Intersect with the frustum-visible mask (if provided).
    5. Compose transforms (texture → scale_N → scale_0 → world) and transform
       bounds back to world coordinates.
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
        world_transform: AffineTransform,
        frustum_visible_chunks: np.ndarray | None = None,
    ) -> ChunkSelectionResult:
        """Select chunks for rendering in a fixed-size texture atlas.

        Parameters
        ----------
        scale_level : ScaleLevelModel
            Scale level with chunk metadata and coordinate transforms.
        view_params : ViewParameters
            Camera state in **world** coordinates.
        texture_config : TextureConfiguration
            Texture size constraints.
        world_transform : AffineTransform
            Transform from scale_0 to world (from
            ``MultiscaleImageModel.world_transform``).  Used to convert view
            parameters into scale_N space and to compose the final
            ``texture_to_world_transform``.
        frustum_visible_chunks : np.ndarray, optional
            Boolean mask (n_chunks,) from frustum culling.  If None all chunks
            are treated as frustum-visible.

        Returns
        -------
        ChunkSelectionResult
        """
        # Step 1: Transform view params world → scale_N
        view_params_scale = self._transform_view_params_to_scale(
            view_params, scale_level, world_transform
        )

        # Step 2: Position texture (anchor = camera-ray projection onto data)
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

        # Step 5: Compose texture→scale_N→scale_0→world and convert bounds
        transform_world = self._compose_transforms(
            world_transform, scale_level.transform, transform_scale
        )
        texture_bounds_world = self._transform_bounds_to_world(
            texture_bounds_scale, scale_level.transform, world_transform
        )

        return ChunkSelectionResult(
            selected_chunk_mask=full_scale_mask,
            texture_to_world_transform=transform_world,
            texture_bounds_world=texture_bounds_world,
            primary_axis=primary_axis,
            n_selected_chunks=n_selected,
        )

    def _transform_view_params_to_scale(
        self,
        view_params: ViewParameters,
        scale_level: ScaleLevelModel,
        world_transform: AffineTransform,
    ) -> ViewParameters:
        """Transform view parameters from world to scale_N coordinates.

        Applies the two-step inverse chain: world → scale_0 (via
        ``world_transform.imap``) then scale_0 → scale_N (via
        ``scale_level.transform.imap``).
        """
        # Frustum corners are points (w=1): world → scale_0 → scale_N
        frustum_flat = view_params.frustum_corners.reshape(-1, 3)
        frustum_flat_0 = world_transform.imap_coordinates(frustum_flat)
        frustum_scale = scale_level.transform.imap_coordinates(frustum_flat_0)
        frustum_corners_scale = frustum_scale.reshape(view_params.frustum_corners.shape)

        # View direction is a vector (w=0) — translation suppressed both steps
        view_dir_0 = world_transform.imap_direction(
            view_params.view_direction.reshape(1, -1)
        ).flatten()
        view_direction_scale = scale_level.transform.imap_direction(
            view_dir_0.reshape(1, -1)
        ).flatten()

        # Near plane center is a point (w=1): world → scale_0 → scale_N
        near_0 = world_transform.imap_coordinates(
            view_params.near_plane_center.reshape(1, -1)
        ).flatten()
        near_scale = scale_level.transform.imap_coordinates(
            near_0.reshape(1, -1)
        ).flatten()

        return ViewParameters(
            frustum_corners=frustum_corners_scale.astype(np.float32),
            view_direction=view_direction_scale.astype(np.float32),
            near_plane_center=near_scale.astype(np.float32),
            canvas_size=view_params.canvas_size,
        )

    def _compose_transforms(
        self,
        world_transform: AffineTransform,
        scale_to_scale0_transform: AffineTransform,
        texture_to_scale_transform: AffineTransform,
    ) -> AffineTransform:
        """Compose texture→scale_N→scale_0→world into a single transform."""
        new_matrix = (
            world_transform.matrix
            @ scale_to_scale0_transform.matrix
            @ texture_to_scale_transform.matrix
        )
        return AffineTransform(matrix=new_matrix)

    def _transform_bounds_to_world(
        self,
        texture_bounds_scale: tuple[np.ndarray, np.ndarray],
        scale_to_scale0_transform: AffineTransform,
        world_transform: AffineTransform,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform texture bounds from scale_N to world coordinates."""
        min_s, max_s = texture_bounds_scale
        # scale_N → scale_0
        corners_0 = scale_to_scale0_transform.map_coordinates(np.vstack([min_s, max_s]))
        # scale_0 → world
        corners_world = world_transform.map_coordinates(corners_0)
        return corners_world[0].astype(np.float32), corners_world[1].astype(np.float32)
