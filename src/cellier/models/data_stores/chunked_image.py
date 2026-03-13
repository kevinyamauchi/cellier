"""Data store for chunked, multiscale 3D images (e.g. zarr/HDF5)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

import numpy as np
from pydantic import ConfigDict, PrivateAttr

from cellier.models.data_stores.base_data_store import BaseDataStore
from cellier.types import (
    ChunkData,
    ChunkedDataRequest,
    ChunkedDataResponse,
    ChunkedSelectedRegion,
    ChunkRequest,
    SceneId,
    SelectedRegion,
    TilingMethod,
    VisualId,
)
from cellier.utils.chunked_image._chunk_culler import ChunkCuller
from cellier.utils.chunked_image._chunk_selection import (
    AxisAlignedTexturePositioning,
    ChunkSelector,
    TextureBoundsFiltering,
    compute_in_view_aabb,
)
from cellier.utils.chunked_image._data_classes import (
    TextureConfiguration,
)
from cellier.utils.chunked_image._multiscale_image_model import (  # noqa: TCH001
    MultiscaleImageModel,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from cellier.slicer.slicer import AsynchronousDataSlicer
    from cellier.utils.chunked_image._data_classes import ViewParameters
    from cellier.utils.chunked_image._multiscale_image_model import ScaleLevelModel


class ChunkedImageStore(BaseDataStore):
    """Data store for chunked, multiscale 3D images backed by zarr or HDF5.

    Wires together :class:`~cellier.utils.chunked_image.ChunkCuller`,
    :class:`~cellier.utils.chunked_image._chunk_selection.ChunkSelector`, and
    :class:`~cellier.utils.chunked_image.ChunkManager` into the
    ``BaseDataStore`` interface consumed by
    :class:`~cellier.slicer.slicer.AsynchronousDataSlicer`.

    Call :meth:`setup_chunk_manager` once after construction to wire up the
    shared slicer thread pool and open the backing array.

    Parameters
    ----------
    multiscale_model : MultiscaleImageModel
        Description of the scale pyramid, including chunk shapes and
        pre-computed chunk corners.
    store_path : str
        Path passed to ``zarr.open`` (or any zarr-compatible store string).
    texture_config : TextureConfiguration
        Texture geometry (cube edge length in voxels) used by the chunk
        selector to determine how many chunks fit in the GPU texture.
    name : str, optional
        Human-readable label for this store.
    """

    multiscale_model: MultiscaleImageModel
    # store_path is kept for backward compatibility; prefer store_paths.
    store_path: str | None = None
    store_paths: list[str] | None = None
    texture_config: TextureConfiguration
    lod_bias: float = 1.0

    # discriminated union key for serialisation
    store_type: Literal["chunked_image"] = "chunked_image"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ------------------------------------------------------------------
    # Private, non-serialised state
    # ------------------------------------------------------------------
    _chunk_manager: Any = PrivateAttr(default=None)
    # One lazily-opened zarr array per scale level (index-matched to
    # multiscale_model.scales / store_paths).
    _zarr_arrays: list[Any] = PrivateAttr(default_factory=list)
    _culler: ChunkCuller = PrivateAttr(default=None)
    _selector: ChunkSelector = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Create stateless strategy objects after Pydantic initialisation."""
        # Normalize store_path (deprecated single-path) → store_paths list.
        if self.store_paths is None:
            if self.store_path is not None:
                self.store_paths = [self.store_path]
            else:
                raise ValueError(
                    "ChunkedImageStore requires either 'store_path' or 'store_paths'."
                )

        self._culler = ChunkCuller()
        self._selector = ChunkSelector(
            positioning_strategy=AxisAlignedTexturePositioning(),
            filtering_strategy=TextureBoundsFiltering(),
        )

    # ------------------------------------------------------------------
    # Public wiring API
    # ------------------------------------------------------------------

    def setup_chunk_manager(
        self,
        slicer: AsynchronousDataSlicer,
        max_cache_bytes: int,
    ) -> None:
        """Open the backing store and wire up the shared thread-pool cache.

        Must be called once before :meth:`get_data_request` / :meth:`get_data`.
        Uses ``slicer._thread_pool`` directly so that chunk I/O shares the same
        workers as all other data requests in the application.

        Parameters
        ----------
        slicer : AsynchronousDataSlicer
            Application-wide async slicer whose thread pool is reused.
        max_cache_bytes : int
            Upper bound on total cached voxel data in bytes.
        """
        from cellier.utils.chunked_image._chunk_manager import ChunkManager

        # _zarr_array is opened lazily on the first _load_chunk call so that
        # tests can inject a pre-built array directly into self._zarr_array
        # without needing a real filesystem path.
        self._chunk_manager = ChunkManager(
            slicer=slicer,
            loader=self._load_chunk,
            max_cache_bytes=max_cache_bytes,
        )

    # ------------------------------------------------------------------
    # BaseDataStore interface
    # ------------------------------------------------------------------

    def get_data_request(
        self,
        selected_region: SelectedRegion,
        tiling_method: TilingMethod,
        visual_id: VisualId,
        scene_id: SceneId,
    ) -> list[ChunkedDataRequest]:
        """Produce chunk requests for the current camera frustum.

        Parameters
        ----------
        selected_region : ChunkedSelectedRegion
            Must be a :class:`~cellier.types.ChunkedSelectedRegion` carrying
            frustum corners, view direction, and near-plane centre in world
            coordinates.
        tiling_method : TilingMethod
            Ignored (chunked stores always tile by frustum).
        visual_id : VisualId
            Visual identifier forwarded into every :class:`~cellier.types.ChunkRequest`.
        scene_id : SceneId
            Scene identifier forwarded into every :class:`~cellier.types.ChunkRequest`.

        Returns
        -------
        list[ChunkedDataRequest]
            A single-element list when chunks are in view, empty when the
            frustum does not intersect the volume.

        Raises
        ------
        TypeError
            If *selected_region* is not a :class:`~cellier.types.ChunkedSelectedRegion`.
        """
        if not isinstance(selected_region, ChunkedSelectedRegion):
            raise TypeError(
                f"ChunkedImageStore requires a ChunkedSelectedRegion, "
                f"got {type(selected_region).__name__}"
            )

        view_params: ViewParameters = selected_region.view_parameters

        # Select the scale based on in-view coverage criterion.
        scale_index = self._select_scale(view_params)
        logger.debug("get_data_request: selected scale_index=%d", scale_index)
        scale_level: ScaleLevelModel = self.multiscale_model.scales[scale_index]
        world_transform = self.multiscale_model.world_transform

        # Propagate the true number of scales so the renderer can allocate
        # one Volume node per scale on first use.
        n_scales = len(self.multiscale_model.scales)
        texture_config = TextureConfiguration(
            texture_width=self.texture_config.texture_width,
            n_scales=n_scales,
        )

        # 1. Frustum culling — boolean mask (n_chunks,)
        frustum_visible_mask = self._culler.cull_chunks(
            scale_level, view_params.frustum_corners, world_transform
        )

        # 2. Texture-bounded chunk selection
        result = self._selector.select_chunks(
            scale_level,
            view_params,
            texture_config,
            world_transform=world_transform,
            frustum_visible_chunks=frustum_visible_mask,
        )
        logger.debug(
            "get_data_request: n_selected_chunks=%d  texture_bounds_world=%s",
            result.n_selected_chunks,
            result.texture_bounds_world,
        )

        if result.n_selected_chunks == 0:
            return []

        # 3. Compute per-chunk priorities (distance from near plane, smaller = closer)
        priorities = self._compute_priorities(
            scale_level, result.selected_chunk_mask, view_params, world_transform
        )

        # 4. Build ChunkRequest list for all selected chunks, including
        #    the (z, y, x) voxel offset of each chunk within the texture atlas.
        grid_shape = scale_level.chunk_grid_shape  # (nz, ny, nx)
        cz, cy, cx = scale_level.chunk_shape
        selected_indices = np.where(result.selected_chunk_mask)[0]
        chunk_requests: list[ChunkRequest] = []
        for idx in selected_indices:
            # Convert linear index → (iz, iy, ix) grid position
            iz = int(idx) // (grid_shape[1] * grid_shape[2])
            iy = (int(idx) // grid_shape[2]) % grid_shape[1]
            ix = int(idx) % grid_shape[2]

            # Min corner of this chunk in scale_n coordinates
            scale_n_corner = np.array([[iz * cz, iy * cy, ix * cx]], dtype=np.float64)
            # Transform scale_N → scale_0 → world, then invert
            # texture_to_world_transform to get the voxel offset in the atlas.
            scale_0_corner = scale_level.transform.map_coordinates(scale_n_corner)
            world_corner = world_transform.map_coordinates(scale_0_corner)
            tex_corner = result.texture_to_world_transform.imap_coordinates(
                world_corner
            )
            texture_offset = tuple(max(0, int(round(v))) for v in tex_corner[0])

            chunk_requests.append(
                ChunkRequest(
                    chunk_index=int(idx),
                    scale_index=scale_index,
                    priority=float(priorities[idx]),
                    visual_id=visual_id,
                    scene_id=scene_id,
                    texture_offset=texture_offset,
                )
            )

        return [
            ChunkedDataRequest(
                scene_id=scene_id,
                visual_id=visual_id,
                resolution_level=scale_index,
                chunk_requests=chunk_requests,
                texture_config=texture_config,
                texture_to_world_transform=result.texture_to_world_transform,
            )
        ]

    def get_data(self, request: ChunkedDataRequest) -> ChunkedDataResponse:
        """Retrieve chunks, returning cached ones immediately.

        Chunks not in the cache are submitted for background loading via the
        shared thread pool; the
        :attr:`~cellier.utils.chunked_image.ChunkManager.chunk_loaded` signal
        fires when each arrives (Phase E wires this to the renderer).

        Parameters
        ----------
        request : ChunkedDataRequest
            The request produced by :meth:`get_data_request`.

        Returns
        -------
        ChunkedDataResponse
            Cached chunks plus a count of chunks still loading in the
            background.
        """
        available: list[ChunkData]
        pending_count: int
        available, pending_count = self._chunk_manager.request_chunks(
            request.chunk_requests, request.visual_id
        )
        return ChunkedDataResponse(
            id=uuid4().hex,
            scene_id=request.scene_id,
            visual_id=request.visual_id,
            resolution_level=request.resolution_level,
            available_chunks=available,
            pending_count=pending_count,
            texture_to_world_transform=request.texture_to_world_transform,
            texture_config=request.texture_config,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_chunk(self, req: ChunkRequest) -> np.ndarray:
        """Synchronously load one chunk from the backing zarr array.

        Called from a background thread; must be thread-safe.  The zarr store
        itself handles concurrent reads safely for most backends.

        Parameters
        ----------
        req : ChunkRequest
            Identifies the chunk by its linear index and scale level.

        Returns
        -------
        np.ndarray
            Voxel data for the requested chunk as float32.
        """
        # Lazily open the backing store for this scale on first access so tests
        # can inject pre-built arrays directly into self._zarr_arrays without
        # needing filesystem access.
        while len(self._zarr_arrays) <= req.scale_index:
            self._zarr_arrays.append(None)

        if self._zarr_arrays[req.scale_index] is None:
            import zarr

            self._zarr_arrays[req.scale_index] = zarr.open(
                self.store_paths[req.scale_index], mode="r"
            )

        zarr_array = self._zarr_arrays[req.scale_index]
        scale_level: ScaleLevelModel = self.multiscale_model.scales[req.scale_index]
        grid_shape = scale_level.chunk_grid_shape  # (nz, ny, nx)

        # Convert row-major linear index → (z, y, x) grid position
        iz = req.chunk_index // (grid_shape[1] * grid_shape[2])
        iy = (req.chunk_index // grid_shape[2]) % grid_shape[1]
        ix = req.chunk_index % grid_shape[2]

        cz, cy, cx = scale_level.chunk_shape
        return np.asarray(
            zarr_array[
                iz * cz : (iz + 1) * cz,
                iy * cy : (iy + 1) * cy,
                ix * cx : (ix + 1) * cx,
            ],
            dtype=np.float32,
        )

    def _select_scale(self, view_params: ViewParameters) -> int:
        """Choose scale using a texture-coverage criterion.

        Iterates from finest (index 0) to coarsest (index N-1) and returns
        the first scale N where the texture (``texture_width * 2^N`` scale_0
        voxels) is large enough to cover the in-view data width:

        ``texture_width * 2^N >= in_view_width * lod_bias``

        ``in_view_width`` is the frustum's perpendicular cross-section width
        **at the depth where the central view ray enters the data**, in scale_0
        coordinates.  Using the entry-point cross-section rather than the full
        frustum AABB prevents the far clipping plane from dominating the metric
        and ensures the selected scale tracks the camera-to-data distance.

        The ``lod_bias`` field shifts the threshold:
        - ``lod_bias < 1``: bias toward finer scales (more detail).
        - ``lod_bias > 1``: bias toward coarser scales (better performance).
        - ``lod_bias = 1``: coverage optimum.

        Falls back to the coarsest scale when the frustum does not intersect
        the data, or when no scale satisfies the criterion.

        Parameters
        ----------
        view_params : ViewParameters
            Current camera state.  ``frustum_corners`` and ``view_direction``
            are used; ``canvas_size`` is not needed for this criterion.

        Returns
        -------
        int
            Scale index to use (0 = finest, N-1 = coarsest).
        """
        n_scales = len(self.multiscale_model.scales)
        world_transform = self.multiscale_model.world_transform

        # Data bounding box in world coordinates.
        data_bb_min_w, data_bb_max_w = self.multiscale_model.get_full_extent_world()

        # Quick overlap check: if full frustum AABB misses the data, bail early.
        if (
            compute_in_view_aabb(
                view_params.frustum_corners, data_bb_min_w, data_bb_max_w
            )
            is None
        ):
            logger.debug(
                "_select_scale: frustum does not overlap data → coarsest scale %d",
                n_scales - 1,
            )
            return n_scales - 1

        # Compute the frustum cross-section at the data-entry depth.
        #
        # This measures the actual zoom level: for a camera at distance d from
        # the data surface with half field-of-view theta, the cross-section is
        # ~2*d*tan(theta), independent of the far clipping distance.
        view_dir = view_params.view_direction.astype(float)
        primary_axis = int(np.argmax(np.abs(view_dir)))
        perp_axes = [ax for ax in range(3) if ax != primary_axis]

        near_corners_w = np.asarray(view_params.frustum_corners[0], dtype=float)
        far_corners_w = np.asarray(view_params.frustum_corners[1], dtype=float)
        near_center_w = near_corners_w.mean(axis=0)
        far_center_w = far_corners_w.mean(axis=0)
        ray_segment_w = far_center_w - near_center_w  # near → far

        # t where the central ray reaches the data face (t=0 at near, t=1 at far).
        data_face_w = (
            data_bb_min_w[primary_axis]
            if view_dir[primary_axis] >= 0
            else data_bb_max_w[primary_axis]
        )
        ray_denom = ray_segment_w[primary_axis]
        if abs(ray_denom) < 1e-10:
            t_entry = 0.0  # ray perpendicular to primary axis → use near plane
        else:
            # Clamp to 0: if data face is behind the near plane, stay at near.
            t_entry = max(
                0.0,
                (data_face_w - near_center_w[primary_axis]) / ray_denom,
            )

        # Interpolate frustum corners to that depth.
        corners_at_entry_w = near_corners_w + t_entry * (far_corners_w - near_corners_w)

        # Clamp cross-section to the data AABB, then convert to scale_0 units.
        cmin_w = np.maximum(corners_at_entry_w.min(axis=0), data_bb_min_w)
        cmax_w = np.minimum(corners_at_entry_w.max(axis=0), data_bb_max_w)
        corners_s0 = world_transform.imap_coordinates(np.vstack([cmin_w, cmax_w]))
        extents_s0 = np.abs(corners_s0[1] - corners_s0[0])
        in_view_width = float(max(extents_s0[ax] for ax in perp_axes))

        logger.debug(
            "_select_scale: t_entry=%.4f  in_view_width_s0=%.2f  primary_axis=%d",
            t_entry,
            in_view_width,
            primary_axis,
        )

        texture_width = float(self.texture_config.texture_width)

        for scale_index in range(n_scales):
            factor = 2.0**scale_index
            coverage = texture_width * factor
            if coverage >= in_view_width * self.lod_bias:
                logger.debug(
                    "_select_scale: selected scale %d "
                    "(coverage=%.1f >= threshold=%.1f)",
                    scale_index,
                    coverage,
                    in_view_width * self.lod_bias,
                )
                return scale_index

        logger.debug(
            "_select_scale: no scale satisfies coverage → coarsest %d",
            n_scales - 1,
        )
        return n_scales - 1

    def _compute_priorities(
        self,
        scale_level: ScaleLevelModel,
        selected_mask: np.ndarray,
        view_params: ViewParameters,
        world_transform: Any,
    ) -> np.ndarray:
        """Compute per-chunk distance from the near plane (lower = render first).

        Parameters
        ----------
        scale_level : ScaleLevelModel
            Scale level whose chunk corners are used to compute centres.
        selected_mask : np.ndarray
            Boolean mask of shape (n_chunks,); only selected entries matter.
        view_params : ViewParameters
            Camera state; supplies ``view_direction`` and ``near_plane_center``.
        world_transform : AffineTransform
            Scale_0 → world transform from :attr:`MultiscaleImageModel.world_transform`.

        Returns
        -------
        np.ndarray
            Signed distances of shape (n_chunks,); unselected entries are
            ``np.inf``.
        """
        n_chunks = scale_level.n_chunks
        priorities = np.full(n_chunks, np.inf, dtype=np.float64)

        selected_indices = np.where(selected_mask)[0]
        if selected_indices.size == 0:
            return priorities

        # Chunk centres in scale_n coords: mean over 8 corners per chunk
        # chunk_corners_scale has shape (n_chunks, 8, 3)
        centers_scale_n = scale_level.chunk_corners_scale[selected_indices].mean(axis=1)

        # Transform scale_n → scale_0 → world (vectorised)
        centers_scale_0 = scale_level.transform.map_coordinates(centers_scale_n)
        centers_world = world_transform.map_coordinates(centers_scale_0)

        # Signed distance from near plane along view direction
        offsets = centers_world - view_params.near_plane_center[np.newaxis, :]
        distances = offsets @ view_params.view_direction

        priorities[selected_indices] = distances
        return priorities
