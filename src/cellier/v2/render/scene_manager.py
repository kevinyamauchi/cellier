"""SceneManager — owns one pygfx Scene and its registered visuals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pygfx as gfx

from cellier.v2.render._scene_config import VisualRenderConfig

if TYPE_CHECKING:
    from uuid import UUID

    from cellier.v2.data.image import ChunkRequest
    from cellier.v2.render._requests import ReslicingRequest


class SceneManager:
    """Owns one pygfx ``gfx.Scene`` and the registry of visuals attached to it.

    Which node (2D or 3D) is active is determined at runtime from the
    ``dims_state`` carried by each ``ReslicingRequest``, not fixed at
    construction time.

    Parameters
    ----------
    scene_id : UUID
        Unique identifier for this scene.
    """

    def __init__(
        self,
        scene_id: UUID,
    ) -> None:
        self._scene_id = scene_id
        self._scene = gfx.Scene()
        self._visuals: dict[UUID, Any] = {}

    @property
    def scene_id(self) -> UUID:
        """Unique identifier for this scene."""
        return self._scene_id

    @property
    def scene(self) -> gfx.Scene:
        """The pygfx Scene object. Passed to CanvasView via get_scene_fn."""
        return self._scene

    @property
    def visual_ids(self) -> list[UUID]:
        """IDs of all registered visuals."""
        return list(self._visuals.keys())

    def add_visual(self, visual: Any, displayed_axes: tuple[int, ...]) -> None:
        """Register a visual and add its node to the scene graph.

        Adds ``node_3d`` if ``len(displayed_axes) == 3``, ``node_2d``
        otherwise.

        Parameters
        ----------
        visual : GFXMultiscaleImageVisual | GFXImageMemoryVisual
            The visual to register.
        displayed_axes : tuple[int, ...]
            Current displayed axes from the scene's dims selection.  Used to
            determine which node to add to the scene graph.

        Raises
        ------
        ValueError
            If the visual does not have a node for this dimensionality.
        """
        if len(displayed_axes) == 3:
            if visual.node_3d is None:
                raise ValueError(
                    f"Visual {visual.visual_model_id} has no 3D node. "
                    "Ensure render_modes includes '3d'."
                )
            self._scene.add(visual.node_3d)
        else:
            if visual.node_2d is None:
                raise ValueError(
                    f"Visual {visual.visual_model_id} has no 2D node. "
                    "Ensure render_modes includes '2d'."
                )
            self._scene.add(visual.node_2d)
        self._visuals[visual.visual_model_id] = visual

    def remove_visual(self, visual_id: UUID) -> None:
        """Unregister a visual and remove its node from the scene graph.

        Parameters
        ----------
        visual_id : UUID
            ID of the visual to remove.
        """
        visual = self._visuals.pop(visual_id)
        if visual.node_3d is not None and visual.node_3d.parent is not None:
            self._scene.remove(visual.node_3d)
        elif visual.node_2d is not None and visual.node_2d.parent is not None:
            self._scene.remove(visual.node_2d)

    def get_visual(self, visual_id: UUID) -> Any:
        """Return the registered visual for ``visual_id``.

        Parameters
        ----------
        visual_id : UUID
            ID of the visual to retrieve.

        Returns
        -------
        GFXMultiscaleImageVisual | GFXImageMemoryVisual

        Raises
        ------
        KeyError
            If ``visual_id`` is not registered in this scene.
        """
        return self._visuals[visual_id]

    def build_slice_requests(
        self,
        request: ReslicingRequest,
        visual_configs: dict[UUID, VisualRenderConfig],
    ) -> dict[UUID, list[ChunkRequest]]:
        """Collect ChunkRequests from all (or targeted) registered visuals.

        Dispatches to the 2D or 3D planning path based on scene
        dimensionality.

        Parameters
        ----------
        request : ReslicingRequest
            The reslicing request.
        visual_configs : dict[UUID, VisualRenderConfig]
            Per-visual render configuration.

        Returns
        -------
        dict[UUID, list[ChunkRequest]]
            Mapping of ``visual_model_id`` to that visual's ChunkRequests.
        """
        if len(request.dims_state.selection.displayed_axes) == 2:
            return self._build_slice_requests_2d(request, visual_configs)
        return self._build_slice_requests_3d(request, visual_configs)

    def _build_slice_requests_3d(
        self,
        request: ReslicingRequest,
        visual_configs: dict[UUID, VisualRenderConfig],
    ) -> dict[UUID, list[ChunkRequest]]:
        """3D planning path using perspective camera and frustum culling."""
        result: dict[UUID, list[ChunkRequest]] = {}
        for visual_id, visual in self._visuals.items():
            if (
                request.target_visual_ids is not None
                and visual_id not in request.target_visual_ids
            ):
                continue

            cfg = visual_configs.get(visual_id, VisualRenderConfig())

            frustum_corners_world = (
                request.frustum_corners if cfg.frustum_cull else None
            )

            n_levels = visual.n_levels
            thresholds = self._compute_thresholds_3d(request, n_levels, cfg.lod_bias)

            chunk_requests = visual.build_slice_request(
                camera_pos_world=request.camera_pos,
                frustum_corners_world=frustum_corners_world,
                thresholds=thresholds,
                dims_state=request.dims_state,
                force_level=cfg.force_level,
            )
            if chunk_requests:
                result[visual_id] = chunk_requests

        return result

    def _build_slice_requests_2d(
        self,
        request: ReslicingRequest,
        visual_configs: dict[UUID, VisualRenderConfig],
    ) -> dict[UUID, list[ChunkRequest]]:
        """2D planning path using orthographic camera and viewport culling."""
        result: dict[UUID, list[ChunkRequest]] = {}

        world_width, world_height = request.world_extent
        viewport_width_px, viewport_height_px = request.screen_size_px

        # Compute viewport AABB from camera position and world extent.
        cx = float(request.camera_pos[0])
        cy = float(request.camera_pos[1])
        half_w = world_width / 2.0
        half_h = world_height / 2.0
        view_min = np.array([cx - half_w, cy - half_h], dtype=np.float64)
        view_max = np.array([cx + half_w, cy + half_h], dtype=np.float64)

        for visual_id, visual in self._visuals.items():
            if (
                request.target_visual_ids is not None
                and visual_id not in request.target_visual_ids
            ):
                continue

            cfg = visual_configs.get(visual_id, VisualRenderConfig())

            chunk_requests = visual.build_slice_request_2d(
                camera_pos_world=request.camera_pos,
                viewport_width_px=viewport_width_px,
                world_width=world_width,
                view_min_world=view_min if cfg.frustum_cull else None,
                view_max_world=view_max if cfg.frustum_cull else None,
                dims_state=request.dims_state,
                lod_bias=cfg.lod_bias,
                force_level=cfg.force_level,
                use_culling=cfg.frustum_cull,
            )
            if chunk_requests:
                result[visual_id] = chunk_requests

        return result

    def _compute_thresholds_3d(
        self, request: ReslicingRequest, n_levels: int, lod_bias: float
    ) -> list[float]:
        """Compute LOD distance thresholds from perspective camera parameters.

        Parameters
        ----------
        request : ReslicingRequest
            The reslicing request providing FOV and screen height.
        n_levels : int
            Number of LOD levels for this visual.
        lod_bias : float
            Multiplier on the computed thresholds.

        Returns
        -------
        list[float]
            One threshold per level boundary (length ``n_levels - 1``).
        """
        _, screen_height_px = request.screen_size_px
        focal_half_height = (screen_height_px / 2.0) / np.tan(request.fov_y_rad / 2.0)
        return [
            (2 ** (k - 1)) * focal_half_height * lod_bias for k in range(1, n_levels)
        ]
