"""Axis-aligned grid-snapped texture positioning strategy.

This module implements the core positioning algorithm that places a cubic texture
in world space by snapping to chunk grid boundaries and aligning with world axes.
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
    """Position texture based on bounding box of frustum chunks.

    This strategy calculates an axis-aligned bounding box of all chunks within
    the view frustum, then selects a positioning corner based on the camera's
    view direction. For each axis, it chooses the minimum coordinate if looking
    in the positive axis direction, or the maximum coordinate if looking in the
    negative axis direction.

    The algorithm prioritizes optimal coverage of chunks closest to the camera
    by positioning the texture to capture the most relevant portion of the
    chunk distribution.
    """

    def position_texture(
        self,
        view_params: ViewParameters,
        scale_level: ScaleLevelModel,
        texture_config: TextureConfiguration,
        frustum_chunk_corners: np.ndarray,
    ) -> tuple[AffineTransform, tuple[np.ndarray, np.ndarray], int]:
        """Position texture based on the chunks in the camera frustum.

        Parameters
        ----------
        view_params : ViewParameters
            Camera view information including view direction vector used for
            axis-specific corner selection.
        scale_level : ScaleLevelModel
            Scale level metadata (unused in this implementation but required
            by the interface).
        texture_config : TextureConfiguration
            Texture configuration settings including texture_width used for
            calculating texture bounds.
        frustum_chunk_corners : np.ndarray
            Array of shape (n_frustum_chunks, 8, 3) containing corner
            coordinates of chunks that are within the view frustum.

        Returns
        -------
        texture_to_world_transform : AffineTransform
            Transform mapping from texture space to world space coordinates
        texture_bounds : tuple[np.ndarray, np.ndarray]
            (texture_min, texture_max) where each is a (3,) array representing
            the world coordinate bounds of the texture
        primary_axis : int
            Index of the primary viewing axis based on
            the largest view direction component
        """
        # Step 1: Calculate bounding box of all frustum chunks
        bbox_min, bbox_max = self._calculate_frustum_chunks_bounding_box(
            frustum_chunk_corners
        )

        # Step 2: Select positioning corner with AABB modification
        # to fit texture constraints
        positioning_corner = self._select_positioning_corner(
            bbox_min, bbox_max, view_params.frustum_corners, texture_config
        )

        # Step 3: Calculate texture bounds with positioning corner as minimum
        texture_bounds = self._calculate_texture_bounds(
            positioning_corner, texture_config
        )

        # Step 4: Create affine transform from texture space to world space
        texture_to_world_transform = AffineTransform.from_translation(
            positioning_corner
        )

        # Step 5: Determine primary axis based on view direction
        primary_axis = self._determine_primary_axis(view_params.view_direction)

        return texture_to_world_transform, texture_bounds, primary_axis

    def _calculate_frustum_chunks_bounding_box(
        self, frustum_chunk_corners: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate axis-aligned bounding box of all chunks in frustum.

        Parameters
        ----------
        frustum_chunk_corners : np.ndarray
            Array of shape (n_frustum_chunks, 8, 3) containing corner
            coordinates for each chunk within the view frustum.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing (bounding_box_min, bounding_box_max) where
            each is a (3,) array representing the minimum and maximum
            coordinates of the axis-aligned bounding box that encompasses
            all frustum chunks.
        """
        # Reshape from (n_chunks, 8, 3) to (n_chunks * 8, 3) to get all corner points
        all_corners = frustum_chunk_corners.reshape(-1, 3)

        # Calculate min and max coordinates across all corners
        bbox_min = np.min(all_corners, axis=0)  # (3,) array
        bbox_max = np.max(all_corners, axis=0)  # (3,) array

        return bbox_min, bbox_max

    def _select_positioning_corner(
        self,
        bounding_box_min: np.ndarray,
        bounding_box_max: np.ndarray,
        frustum_corners: np.ndarray,
        texture_config: TextureConfiguration,
    ) -> np.ndarray:
        """Select positioning corner by modifying AABB to fit texture constraints.

        This method finds the face of the axis-aligned bounding box closest to
        the camera, identifies the vertex on that face closest to the camera,
        and then shrinks any oversized edges by moving vertices away from the
        fixed vertex to ensure the modified AABB fits within texture dimensions.

        Parameters
        ----------
        bounding_box_min : np.ndarray
            Array of shape (3,) containing minimum coordinates of the
            original bounding box in (x, y, z) order.
        bounding_box_max : np.ndarray
            Array of shape (3,) containing maximum coordinates of the
            original bounding box in (x, y, z) order.
        frustum_corners : np.ndarray
            Array of shape (2, 4, 3) containing frustum corner coordinates
            where frustum_corners[0] represents the near plane corners.
        texture_config : TextureConfiguration
            Configuration object containing texture_width which defines
            the maximum edge length for the texture.

        Returns
        -------
        np.ndarray
            Array of shape (3,) containing the positioning corner coordinates
            (minimum corner of the modified bounding box that fits within
            texture constraints).
        """
        # Step 1: Calculate camera position as centroid of near plane
        near_plane_centroid = np.mean(frustum_corners[0], axis=0)  # (3,)

        # Step 2: Find the face of the AABB closest to camera
        closest_face_idx = self._find_closest_face_to_camera(
            bounding_box_min, bounding_box_max, near_plane_centroid
        )

        # Step 3: Get vertices on the closest face and find the one closest to camera
        face_vertices = self._get_face_vertices(
            bounding_box_min, bounding_box_max, closest_face_idx
        )
        fixed_vertex = self._find_closest_vertex_to_camera(
            face_vertices, near_plane_centroid
        )

        # Step 4: Modify AABB by shortening edges that exceed texture width
        modified_bbox_min, modified_bbox_max = self._shorten_oversized_edges(
            bounding_box_min,
            bounding_box_max,
            fixed_vertex,
            texture_config.texture_width,
        )

        return modified_bbox_min

    def _find_closest_face_to_camera(
        self, bbox_min: np.ndarray, bbox_max: np.ndarray, camera_position: np.ndarray
    ) -> int:
        """Find which face of the AABB is closest to the camera.

        Parameters
        ----------
        bbox_min : np.ndarray
            Minimum coordinates of the bounding box, shape (3,).
        bbox_max : np.ndarray
            Maximum coordinates of the bounding box, shape (3,).
        camera_position : np.ndarray
            Camera position coordinates, shape (3,).

        Returns
        -------
        int
            Index of the closest face (0-5, representing the 6 faces of the AABB).
        """
        # Calculate centers of all 6 faces of the AABB
        face_centers = np.array(
            [
                [
                    bbox_min[0],
                    (bbox_min[1] + bbox_max[1]) / 2,
                    (bbox_min[2] + bbox_max[2]) / 2,
                ],  # -X face
                [
                    bbox_max[0],
                    (bbox_min[1] + bbox_max[1]) / 2,
                    (bbox_min[2] + bbox_max[2]) / 2,
                ],  # +X face
                [
                    (bbox_min[0] + bbox_max[0]) / 2,
                    bbox_min[1],
                    (bbox_min[2] + bbox_max[2]) / 2,
                ],  # -Y face
                [
                    (bbox_min[0] + bbox_max[0]) / 2,
                    bbox_max[1],
                    (bbox_min[2] + bbox_max[2]) / 2,
                ],  # +Y face
                [
                    (bbox_min[0] + bbox_max[0]) / 2,
                    (bbox_min[1] + bbox_max[1]) / 2,
                    bbox_min[2],
                ],  # -Z face
                [
                    (bbox_min[0] + bbox_max[0]) / 2,
                    (bbox_min[1] + bbox_max[1]) / 2,
                    bbox_max[2],
                ],  # +Z face
            ]
        )

        # Calculate distances from camera to each face center
        distances = np.linalg.norm(face_centers - camera_position, axis=1)

        # Return index of face with minimum distance
        return int(np.argmin(distances))

    def _get_face_vertices(
        self, bbox_min: np.ndarray, bbox_max: np.ndarray, face_idx: int
    ) -> np.ndarray:
        """Get the 4 vertices that form a specific face of the AABB.

        Parameters
        ----------
        bbox_min : np.ndarray
            Minimum coordinates of the bounding box, shape (3,).
        bbox_max : np.ndarray
            Maximum coordinates of the bounding box, shape (3,).
        face_idx : int
            Index of the face (0-5).

        Returns
        -------
        np.ndarray
            Array of shape (4, 3) containing the vertices of the specified face.
        """
        if face_idx == 0:  # -X face
            return np.array(
                [
                    [bbox_min[0], bbox_min[1], bbox_min[2]],
                    [bbox_min[0], bbox_min[1], bbox_max[2]],
                    [bbox_min[0], bbox_max[1], bbox_min[2]],
                    [bbox_min[0], bbox_max[1], bbox_max[2]],
                ]
            )
        elif face_idx == 1:  # +X face
            return np.array(
                [
                    [bbox_max[0], bbox_min[1], bbox_min[2]],
                    [bbox_max[0], bbox_min[1], bbox_max[2]],
                    [bbox_max[0], bbox_max[1], bbox_min[2]],
                    [bbox_max[0], bbox_max[1], bbox_max[2]],
                ]
            )
        elif face_idx == 2:  # -Y face
            return np.array(
                [
                    [bbox_min[0], bbox_min[1], bbox_min[2]],
                    [bbox_min[0], bbox_min[1], bbox_max[2]],
                    [bbox_max[0], bbox_min[1], bbox_min[2]],
                    [bbox_max[0], bbox_min[1], bbox_max[2]],
                ]
            )
        elif face_idx == 3:  # +Y face
            return np.array(
                [
                    [bbox_min[0], bbox_max[1], bbox_min[2]],
                    [bbox_min[0], bbox_max[1], bbox_max[2]],
                    [bbox_max[0], bbox_max[1], bbox_min[2]],
                    [bbox_max[0], bbox_max[1], bbox_max[2]],
                ]
            )
        elif face_idx == 4:  # -Z face
            return np.array(
                [
                    [bbox_min[0], bbox_min[1], bbox_min[2]],
                    [bbox_min[0], bbox_max[1], bbox_min[2]],
                    [bbox_max[0], bbox_min[1], bbox_min[2]],
                    [bbox_max[0], bbox_max[1], bbox_min[2]],
                ]
            )
        else:  # face_idx == 5, +Z face
            return np.array(
                [
                    [bbox_min[0], bbox_min[1], bbox_max[2]],
                    [bbox_min[0], bbox_max[1], bbox_max[2]],
                    [bbox_max[0], bbox_min[1], bbox_max[2]],
                    [bbox_max[0], bbox_max[1], bbox_max[2]],
                ]
            )

    def _find_closest_vertex_to_camera(
        self, face_vertices: np.ndarray, camera_position: np.ndarray
    ) -> np.ndarray:
        """Find the vertex on the face that is closest to the camera.

        Parameters
        ----------
        face_vertices : np.ndarray
            Array of shape (4, 3) containing the vertices of a face.
        camera_position : np.ndarray
            Camera position coordinates, shape (3,).

        Returns
        -------
        np.ndarray
            Coordinates of the closest vertex, shape (3,).
        """
        # Calculate distances from camera to each vertex
        distances = np.linalg.norm(face_vertices - camera_position, axis=1)

        # Return vertex with minimum distance
        closest_idx = np.argmin(distances)
        return face_vertices[closest_idx]

    def _shorten_oversized_edges(
        self,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        fixed_vertex: np.ndarray,
        texture_width: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shorten AABB edges that exceed texture width.

        This moves far vertices toward the fixed vertex.
        The fixed vertex is the vertex on the face closest to the camera
        that is closest to the camera.

        Parameters
        ----------
        bbox_min : np.ndarray
            Original minimum coordinates of the bounding box, shape (3,).
        bbox_max : np.ndarray
            Original maximum coordinates of the bounding box, shape (3,).
        fixed_vertex : np.ndarray
            Coordinates of the vertex that should remain unchanged, shape (3,).
        texture_width : float
            Maximum allowed edge length for the texture.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Modified (bbox_min, bbox_max) where each is shape (3,) and
            all edges are at most texture_width in length.
        """
        # Start with original bounds
        modified_min = bbox_min.copy()
        modified_max = bbox_max.copy()

        # For each axis, ensure the edge length doesn't exceed texture_width
        for axis in range(3):
            # Calculate current edge length along this axis
            edge_length = bbox_max[axis] - bbox_min[axis]

            if edge_length > texture_width:
                # Edge is too long, need to shorten it
                # Move the far vertex toward the fixed vertex
                if np.abs(fixed_vertex[axis] - bbox_min[axis]) < np.abs(
                    fixed_vertex[axis] - bbox_max[axis]
                ):
                    # Fixed vertex is closer to bbox_min,
                    # so move bbox_max toward fixed vertex
                    modified_max[axis] = fixed_vertex[axis] + texture_width
                else:
                    # Fixed vertex is closer to bbox_max,
                    # so move bbox_min toward fixed vertex
                    modified_min[axis] = fixed_vertex[axis] - texture_width

        return modified_min, modified_max

    def _calculate_texture_bounds(
        self, positioning_corner: np.ndarray, texture_config: TextureConfiguration
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate world coordinates of axis-aligned texture bounds.

        Creates texture bounds as an axis-aligned cube with the positioning
        corner as the minimum corner and extending by texture_width in each
        dimension.

        Parameters
        ----------
        positioning_corner : np.ndarray
            Array of shape (3,) containing the world coordinates of the
            texture's minimum corner (positioning point).
        texture_config : TextureConfiguration
            Configuration object containing texture_width which defines
            the edge length of the cubic texture in world units.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing (texture_min, texture_max) where each is a
            (3,) array representing the minimum and maximum world coordinates
            of the axis-aligned texture bounds.
        """
        texture_min = positioning_corner.copy()

        # Create texture_max by adding texture width to each dimension
        texture_size = np.array(
            [
                texture_config.texture_width,
                texture_config.texture_width,
                texture_config.texture_width,
            ],
            dtype=np.float32,
        )

        texture_max = positioning_corner + texture_size

        return texture_min, texture_max

    def _determine_primary_axis(self, view_direction: np.ndarray) -> int:
        """Determine primary viewing axis based on largest view direction component.

        Identifies which coordinate axis (X, Y, or Z) has the largest absolute
        component in the view direction vector. This axis represents the
        primary direction of camera viewing.

        Parameters
        ----------
        view_direction : np.ndarray
            Normalized view direction vector of shape (3,) representing
            the camera's viewing direction in world coordinates.

        Returns
        -------
        int
            Index of the primary viewing axis: 0 for X-axis, 1 for Y-axis,
            or 2 for Z-axis.
        """
        # Find axis with largest absolute component
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

        Determines which candidate chunks should be selected by testing whether
        all 8 corners of each chunk fall within the texture bounds. This ensures
        that selected chunks can be completely packed into the texture without
        overlap or partial inclusion.

        Parameters
        ----------
        chunk_indices : np.ndarray
            Array of shape (n_candidate_chunks,) containing linear indices
            for candidate chunks. These correspond to the chunks being tested.
        chunk_corners_world : np.ndarray
            Array of shape (n_candidate_chunks, 8, 3) containing the corner
            coordinates of each candidate chunk in world space. Each chunk
            has 8 corners defining its bounding box, with coordinates in
            (z, y, x) order.
        texture_bounds_world : tuple[np.ndarray, np.ndarray]
            Texture bounds in world coordinates as (min_corner, max_corner)
            where each corner is a (3,) array containing (z, y, x) coordinates.
            These bounds define the 3D region covered by the texture.
        view_params : ViewParameters
            Camera view information (not used in this filtering strategy but
            required by the interface for consistency with other strategies).
        texture_config : TextureConfiguration
            Texture configuration settings (not used in this filtering strategy
            but required by the interface).

        Returns
        -------
        np.ndarray
            Boolean array of shape (n_candidate_chunks,) where True indicates
            that the corresponding chunk should be selected (all corners within
            bounds) and False indicates exclusion.
        """
        # Handle empty input case
        if chunk_indices.size == 0:
            return np.array([], dtype=bool)

        # Extract texture bounds
        texture_min, texture_max = texture_bounds_world

        # Test which chunks fit completely within bounds using vectorized operations
        fitting_mask = self._chunks_within_bounds_vectorized(
            chunk_corners_world, texture_min, texture_max
        )

        return fitting_mask

    def _chunks_within_bounds_vectorized(
        self,
        chunk_corners_world: np.ndarray,
        texture_min: np.ndarray,
        texture_max: np.ndarray,
    ) -> np.ndarray:
        """Test which chunks are completely within texture bounds using vectorization.

        Uses numpy broadcasting to efficiently test all chunk corners against
        texture bounds simultaneously. A chunk is considered "within bounds"
        only if all 8 of its corner points satisfy the bounds constraints.

        Parameters
        ----------
        chunk_corners_world : np.ndarray
            Array of shape (n_chunks, 8, 3) containing corner coordinates
            for all candidate chunks in world space.
        texture_min : np.ndarray
            Array of shape (3,) containing minimum bounds of texture in
            world coordinates as (z, y, x).
        texture_max : np.ndarray
            Array of shape (3,) containing maximum bounds of texture in
            world coordinates as (z, y, x).

        Returns
        -------
        np.ndarray
            Boolean array of shape (n_chunks,) indicating which chunks
            have all corners within the texture bounds.

        Notes
        -----
        The vectorized algorithm:
        1. Uses broadcasting to compare all corners against min/max bounds
        2. Creates boolean arrays for min and max constraint satisfaction
        3. Combines constraints with logical AND operation
        4. Reduces over corner and coordinate dimensions to get per-chunk results

        Broadcasting pattern:
        - chunk_corners_world: (n_chunks, 8, 3)
        - texture_min: (3,) -> broadcast to (1, 1, 3)
        - texture_max: (3,) -> broadcast to (1, 1, 3)
        - Result: (n_chunks, 8, 3) boolean arrays

        Performance characteristics:
        - O(n_chunks * 8 * 3) element-wise comparisons
        - Vectorized operations minimize Python overhead
        - Memory usage scales linearly with number of chunks
        """
        # Use numpy broadcasting to compare all corners against bounds simultaneously
        # chunk_corners_world shape: (n_chunks, 8, 3)
        # texture_min/max shape: (3,) -> broadcasts to (1, 1, 3)

        # Test if all corners are >= texture_min
        within_min = chunk_corners_world >= texture_min[None, None, :]

        # Test if all corners are <= texture_max
        within_max = chunk_corners_world <= texture_max[None, None, :]

        # Combine constraints: corners must satisfy both min AND max bounds
        within_bounds = within_min & within_max

        # A chunk is selected only if ALL of its corners are within bounds
        # Reduce over corner dimension (axis=1) and coordinate dimension (axis=2)
        chunk_fits = np.all(within_bounds, axis=(1, 2))

        return chunk_fits


class ChunkSelector:
    """Main coordinator for chunk selection algorithms using strategy pattern.

    The ChunkSelector orchestrates the entire chunk selection process by combining
    positioning and filtering strategies. It coordinates the workflow of:
    1. Determining candidate chunks from frustum filtering
    2. Transforming view parameters to scale coordinates for efficiency
    3. Positioning the texture in scale coordinate space
    4. Filtering chunks based on texture bounds
    5. Assembling results with proper coordinate transformations

    The class uses the strategy pattern to allow different positioning and filtering
    algorithms to be composed at runtime while maintaining a consistent interface.
    All processing is done in scale coordinates for efficiency, with final results
    transformed back to world coordinates.
    """

    def __init__(
        self,
        positioning_strategy: TexturePositioningStrategy,
        filtering_strategy: ChunkFilteringStrategy,
    ):
        """Initialize chunk selector with positioning and filtering strategies.

        Parameters
        ----------
        positioning_strategy : TexturePositioningStrategy
            Strategy implementation for positioning texture in coordinate space.
            Determines where to place the fixed-size texture for optimal chunk capture.
        filtering_strategy : ChunkFilteringStrategy
            Strategy implementation for filtering chunks based on spatial criteria.
            Determines which chunks should be included in the final selection.
        """
        self._positioning_strategy = positioning_strategy
        self._filtering_strategy = filtering_strategy

    def select_chunks(
        self,
        scale_level: ScaleLevelModel,
        view_params: ViewParameters,
        texture_config: TextureConfiguration,
        frustum_visible_chunks: np.ndarray | None = None,
    ) -> ChunkSelectionResult:
        """Select chunks for rendering in fixed-size texture.

        Coordinates the complete chunk selection process by combining positioning
        and filtering strategies. The algorithm works in scale coordinates for
        efficiency, transforming only the minimal necessary data between coordinate
        systems.

        Parameters
        ----------
        scale_level : ScaleLevelModel
            Scale level containing chunk metadata, coordinate information, and
            transformation data. Provides chunk corners in both scale and world
            coordinates along with the transformation between them.
        view_params : ViewParameters
            Camera view information including frustum corners, camera position,
            view direction, and near plane center. Used for texture positioning
            and chunk prioritization.
        texture_config : TextureConfiguration
            Texture configuration settings including texture dimensions and
            optional chunk limits. Controls the size and constraints of the
            texture used for chunk storage.
        frustum_visible_chunks : np.ndarray, optional
            Boolean array of shape (n_chunks,) indicating which chunks are
            visible within the camera frustum. If None, all chunks in the
            scale level are considered as candidates. This provides an optional
            pre-filtering step to reduce computational overhead.

        Returns
        -------
        ChunkSelectionResult
            Complete selection result containing the boolean mask of selected
            chunks, texture-to-world transformation, texture bounds in world
            coordinates, primary viewing axis, and convenience metadata.

        Notes
        -----
        The algorithm is optimized for performance by:
        - Working in scale coordinates to avoid transforming many chunk corners
        - Transforming only view parameters (small, fixed cost) to scale space
        - Using candidate chunk indices to process only relevant chunks
        - Avoiding expensive mask conversions with direct index operations

        Coordinate system handling:
        1. Transform view parameters from world to scale coordinates
        2. Perform positioning and filtering in scale coordinates
        3. Transform final results back to world coordinates for output

        This approach minimizes coordinate transformations while maintaining
        correct spatial relationships throughout the selection process.
        """
        # Step 1: Determine candidate chunk indices from frustum filtering
        if frustum_visible_chunks is not None:
            candidate_indices = np.flatnonzero(frustum_visible_chunks)
        else:
            candidate_indices = np.arange(scale_level.n_chunks)

        if candidate_indices.size == 0:
            # If no chunks in frustum, return empty result
            identity_transform = AffineTransform.from_translation(np.zeros(3))
            zero_bounds = (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))

            return ChunkSelectionResult(
                selected_chunk_mask=np.zeros(scale_level.n_chunks, dtype=bool),
                texture_to_world_transform=identity_transform,
                texture_bounds_world=zero_bounds,
                primary_axis=0,
                n_selected_chunks=0,
            )

        # Step 2: Transform view parameters to scale coordinates for efficiency
        view_params_scale = self._transform_view_params_to_scale(
            view_params, scale_level
        )

        # Step 3: Position texture in scale coordinates
        transform_scale, texture_bounds_scale, primary_axis = (
            self._positioning_strategy.position_texture(
                view_params_scale,
                scale_level,
                texture_config,
                scale_level.chunk_corners_scale[frustum_visible_chunks],
            )
        )

        # Step 4: Extract candidate chunk corners (already in scale coordinates)
        candidate_corners = scale_level.chunk_corners_scale[candidate_indices]

        # Step 5: Filter chunks based on texture bounds in scale coordinates
        selected_mask = self._filtering_strategy.filter_chunks(
            candidate_indices,
            candidate_corners,
            texture_bounds_scale,
            view_params_scale,
            texture_config,
        )

        # Step 6: Build full-scale boolean mask efficiently
        full_scale_mask = np.zeros(scale_level.n_chunks, dtype=bool)
        if np.any(selected_mask):
            full_scale_mask[candidate_indices[selected_mask]] = True

        # Step 7: Transform results back to world coordinates for output
        transform_world = self._compose_transforms(
            scale_level.transform, transform_scale
        )
        texture_bounds_world = self._transform_bounds_to_world(
            texture_bounds_scale, scale_level.transform
        )

        # Step 8: Construct and return result
        return ChunkSelectionResult(
            selected_chunk_mask=full_scale_mask,
            texture_to_world_transform=transform_world,
            texture_bounds_world=texture_bounds_world,
            primary_axis=primary_axis,
            n_selected_chunks=np.sum(full_scale_mask),
        )

    def _transform_view_params_to_scale(
        self, view_params: "ViewParameters", scale_level: ScaleLevelModel
    ) -> "ViewParameters":
        """Transform view parameters from world coordinates to scale coordinates.

        Applies the inverse of the scale-to-world transformation to convert all
        spatial information in the view parameters to the scale coordinate system.
        This enables positioning and filtering algorithms to work directly with
        chunk coordinates without expensive coordinate conversions.

        Parameters
        ----------
        view_params : ViewParameters
            View parameters in world coordinates to be transformed.
        scale_level : ScaleLevelModel
            Scale level containing the transformation from scale to world coordinates.

        Returns
        -------
        ViewParameters
            New ViewParameters object with all spatial data transformed to
            scale coordinates. Maintains the same structure as input with
            updated coordinate values.
        """
        # Transform frustum corners
        frustum_corners_flat = view_params.frustum_corners.reshape(-1, 3)
        frustum_corners_scale_flat = scale_level.transform.imap_coordinates(
            frustum_corners_flat
        )
        frustum_corners_scale = frustum_corners_scale_flat.reshape(
            view_params.frustum_corners.shape
        )

        # Transform view direction
        view_direction_scale = scale_level.transform.imap_coordinates(
            view_params.view_direction.reshape(1, -1)
        ).flatten()

        # Transform near plane center
        near_plane_center_scale = scale_level.transform.imap_coordinates(
            view_params.near_plane_center.reshape(1, -1)
        ).flatten()

        return ViewParameters(
            frustum_corners=frustum_corners_scale.astype(np.float32),
            view_direction=view_direction_scale.astype(np.float32),
            near_plane_center=near_plane_center_scale.astype(np.float32),
        )

    def _compose_transforms(self, scale_to_world_transform, texture_to_scale_transform):
        """Compose texture-to-scale and scale-to-world transforms.

        Creates the final texture-to-world transformation by composing the
        texture positioning transform (in scale coordinates) with the scale's
        transformation to world coordinates.

        Parameters
        ----------
        scale_to_world_transform : AffineTransform
            Transformation from scale coordinates to world coordinates.
        texture_to_scale_transform : AffineTransform
            Transformation from texture coordinates to scale coordinates.

        Returns
        -------
        AffineTransform
            Composed transformation from texture coordinates to world coordinates.
        """
        # Compose transformations: texture -> scale -> world
        new_matrix = scale_to_world_transform.matrix @ texture_to_scale_transform.matrix
        return AffineTransform(matrix=new_matrix)

    def _transform_bounds_to_world(
        self,
        texture_bounds_scale: tuple[np.ndarray, np.ndarray],
        scale_to_world_transform,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform texture bounds from scale coordinates to world coordinates.

        Applies the scale-to-world transformation to convert texture boundary
        coordinates for use in the final result. The bounds define the 3D region
        of world space covered by the texture.

        Parameters
        ----------
        texture_bounds_scale : tuple[np.ndarray, np.ndarray]
            Texture bounds in scale coordinates as (min_corner, max_corner).
        scale_to_world_transform : AffineTransform
            Transformation from scale coordinates to world coordinates.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Texture bounds in world coordinates as (min_corner, max_corner).
        """
        min_corner_scale, max_corner_scale = texture_bounds_scale

        # Transform both corners to world coordinates
        corners_scale = np.vstack([min_corner_scale, max_corner_scale])
        corners_world = scale_to_world_transform.map_coordinates(corners_scale)

        min_corner_world = corners_world[0].astype(np.float32)
        max_corner_world = corners_world[1].astype(np.float32)

        return min_corner_world, max_corner_world
