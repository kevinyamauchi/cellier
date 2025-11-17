"""Abstract base classes for chunk selection strategies.

This module defines the interfaces for texture positioning and chunk filtering
strategies used in the chunked 3D image rendering system.
"""

from abc import ABC, abstractmethod

import numpy as np

from cellier.utils.chunked_image._data_classes import (
    TextureConfiguration,
    ViewParameters,
)
from cellier.utils.chunked_image._multiscale_image_model import ScaleLevelModel


class TexturePositioningStrategy(ABC):
    """Abstract base class for texture positioning algorithms.

    Texture positioning strategies determine where to place a fixed-size texture
    in world space based on camera view parameters. The texture is positioned to
    optimally capture visible chunks for rendering.
    """

    @abstractmethod
    def position_texture(
        self,
        view_params: ViewParameters,
        scale_level: ScaleLevelModel,
        texture_config: TextureConfiguration,
        frustum_chunk_corners: np.ndarray,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], int]:
        """Position texture in world space based on view parameters.

        This method determines the optimal placement of a cubic texture in world
        coordinates to capture the most relevant chunks for the current view.

        Parameters
        ----------
        view_params : ViewParameters
            Camera view information including frustum corners, camera position,
            view direction, and near plane center.
        scale_level : ScaleLevelModel
            Scale level metadata including chunk shape, transforms, and
            pre-computed chunk corner coordinates.
        texture_config : TextureConfiguration
            Texture configuration settings including texture width and
            optional maximum chunk limit.
        frustum_chunk_corners : np.ndarray
            Array of shape (n_frustum_chunks, 8, 3) containing corner
            coordinates of chunks that are within the view frustum.


        Returns
        -------
        texture_to_world_transform : np.ndarray
            (4, 4) transformation matrix from texture coordinates to world
            coordinates. Texture coordinates range from (0,0,0) to
            (texture_width, texture_width, texture_width).
        texture_bounds_world : tuple[np.ndarray, np.ndarray]
            Texture bounds in world coordinates as (min_corner, max_corner)
            where each corner is a (3,) array containing (z, y, x) coordinates.
        primary_axis : int
            Primary viewing axis used for texture orientation.
            0=X, 1=Y, 2=Z corresponding to the axis with the largest
            component in the view direction.

        Raises
        ------
        ValueError
            If input parameters are invalid (e.g., invalid array shapes,
            non-normalized view direction, negative texture width).
        RuntimeError
            If the positioning algorithm fails to compute a valid texture
            placement (e.g., numerical instability, degenerate view parameters).

        Notes
        -----
        Implementations should:
        - Position texture to maximize coverage of visible chunks
        - Ensure texture bounds are properly aligned with chunk boundaries
        - Handle edge cases like camera inside volume or extreme viewing angles
        - Maintain consistent coordinate system conventions (z, y, x ordering)
        """
        pass


class ChunkFilteringStrategy(ABC):
    """Abstract base class for chunk filtering algorithms.

    Chunk filtering strategies determine which chunks should be included in
    the texture based on spatial criteria such as intersection with texture
    bounds, distance from camera, or other geometric constraints.
    """

    @abstractmethod
    def filter_chunks(
        self,
        chunk_indices: np.ndarray,
        chunk_corners_world: np.ndarray,
        texture_bounds_world: tuple[np.ndarray, np.ndarray],
        view_params: "ViewParameters",
        texture_config: "TextureConfiguration",
    ) -> np.ndarray:
        """Filter chunks based on spatial criteria.

        This method determines which of the candidate chunks should be included
        in the final selection based on geometric and spatial constraints.

        Parameters
        ----------
        chunk_indices : np.ndarray
            (n_candidate_chunks,) array of linear indices for candidate chunks.
            These are indices into the scale level's chunk arrays.
        chunk_corners_world : np.ndarray
            (n_candidate_chunks, 8, 3) array of chunk corner coordinates in
            world space. Each chunk has 8 corners defining its bounding box,
            with coordinates in (z, y, x) order.
        texture_bounds_world : tuple[np.ndarray, np.ndarray]
            Texture bounds in world coordinates as (min_corner, max_corner)
            where each corner is a (3,) array containing (z, y, x) coordinates.
        view_params : ViewParameters
            Camera view information including frustum corners, camera position,
            view direction, and near plane center.
        texture_config : TextureConfiguration
            Texture configuration settings including texture width and
            optional maximum chunk limit.

        Returns
        -------
        selected_mask : np.ndarray
            (n_candidate_chunks,) boolean mask indicating which chunks are
            selected. True values correspond to chunks that should be included
            in the final texture.

        Raises
        ------
        ValueError
            If input parameters are invalid (e.g., mismatched array shapes,
            invalid texture bounds, empty chunk arrays).
        RuntimeError
            If the filtering algorithm fails to complete (e.g., numerical
            issues, unexpected geometric configurations).

        Notes
        -----
        Implementations should:
        - Handle empty candidate chunk arrays gracefully
        - Respect texture_config.max_chunks if specified
        - Prioritize chunks based on relevance to current view
        - Ensure selected chunks can fit within the texture bounds
        - Maintain performance for large numbers of candidate chunks
        """
        pass
