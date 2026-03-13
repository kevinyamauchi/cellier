"""Data store for chunked, multiscale 3D images (e.g. zarr/HDF5)."""

from __future__ import annotations

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
)
from cellier.utils.chunked_image._data_classes import (
    TextureConfiguration,
)
from cellier.utils.chunked_image._multiscale_image_model import (  # noqa: TCH001
    MultiscaleImageModel,
)

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

        # Select the coarsest scale whose visible chunks still fit in the
        # texture atlas.  Falls back to the coarsest scale if none qualifies.
        scale_index = self._select_scale(view_params)
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
        """Choose scale using a mip-map criterion on physical pixel size.

        Iterates from finest (index 0) to coarsest (index N-1) and returns
        the first scale where one data voxel is at least as large as one
        screen pixel (in world-space units).  This avoids loading unnecessary
        detail when the camera is zoomed out.

        The ``lod_bias`` field shifts the threshold:
        - ``lod_bias < 1``: bias toward finer scales (more detail).
        - ``lod_bias > 1``: bias toward coarser scales (better performance).
        - ``lod_bias = 1``: mip-map optimum — one voxel per screen pixel.

        For **perspective** cameras the near clipping plane is typically very
        close to the camera (PyGFX default: 0.1), making its physical height far
        too small to use directly.  Instead the camera position is recovered from
        the ratio of near- and far-plane heights, and the pixel footprint is
        evaluated at the center of the data bounding box.  This correctly
        reflects how large data voxels appear on screen regardless of the near-
        clip distance.

        For **orthographic** cameras the near- and far-plane heights are equal,
        and the near-plane height divided by ``canvas_size[1]`` is used directly.

        Falls back to the coarsest scale if no scale satisfies the criterion
        (e.g., when ``canvas_size`` is (1, 1) — the default placeholder).

        Parameters
        ----------
        view_params : ViewParameters
            Current camera state.  ``frustum_corners`` and ``canvas_size``
            are used to derive world-space-per-pixel.

        Returns
        -------
        int
            Scale index to use (0 = finest, N-1 = coarsest).
        """
        near_corners = view_params.frustum_corners[0]  # (4, 3)
        far_corners = view_params.frustum_corners[1]  # (4, 3)
        canvas_h = view_params.canvas_size[1]
        n_scales = len(self.multiscale_model.scales)

        # corner order: left-bottom, right-bottom, right-top, left-top
        near_h = float(np.linalg.norm(near_corners[3] - near_corners[0]))
        far_h = float(np.linalg.norm(far_corners[3] - far_corners[0]))

        if canvas_h <= 1 or near_h <= 0:
            return n_scales - 1  # coarsest fallback

        if far_h <= near_h * 1.001:
            # Orthographic camera: pixel size is uniform across depth
            world_per_pixel = near_h / canvas_h
        else:
            # Perspective camera: the near plane is typically very close to the
            # camera (e.g. default near=0.1 in PyGFX), so using near_h directly
            # gives a world_per_pixel that is far too small.  Instead, recover the
            # camera position and evaluate pixel footprint at the data center.
            #
            # For a symmetric perspective frustum the ratio of plane heights equals
            # the ratio of plane distances from the camera:
            #   near_h / far_h = near_dist / far_dist
            # Using plane centers and the view direction:
            near_center = near_corners.mean(axis=0)
            far_center = far_corners.mean(axis=0)
            view_dir = view_params.view_direction  # unit vector

            k = far_h / near_h  # k > 1
            # Distance from near plane to far plane along view direction:
            d_near_far = float(np.dot(far_center - near_center, view_dir))
            if d_near_far <= 0:
                return n_scales - 1  # degenerate frustum

            # near_dist = d_near_far / (k - 1)
            near_dist = d_near_far / (k - 1.0)
            cam_pos = near_center - near_dist * view_dir

            # Data center in world coordinates (scale_0 coords = world for
            # default identity world_transform)
            data_bb_min, data_bb_max = self.multiscale_model.get_full_extent_world()
            data_center_world = (data_bb_min + data_bb_max) / 2.0

            dist_to_data = float(np.dot(data_center_world - cam_pos, view_dir))
            if dist_to_data <= 0:
                dist_to_data = near_dist  # fallback: evaluate at near plane

            # World height at the data plane scales linearly with distance
            world_h_at_data = near_h * dist_to_data / near_dist
            world_per_pixel = world_h_at_data / canvas_h

        for scale_index in range(n_scales):
            voxel_size = 2.0**scale_index
            if voxel_size >= world_per_pixel * self.lod_bias:
                return scale_index

        return n_scales - 1  # coarsest fallback

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
