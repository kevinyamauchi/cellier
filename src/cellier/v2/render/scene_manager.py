"""SceneManager — owns one pygfx Scene and its registered visuals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygfx as gfx

from cellier.v2.render._frustum import frustum_planes_from_corners
from cellier.v2.render._scene_config import SceneRenderConfig

if TYPE_CHECKING:
    from uuid import UUID

    from cellier.v2.data.image import ChunkRequest
    from cellier.v2.render._requests import ReslicingRequest
    from cellier.v2.render.visuals._image import GFXMultiscaleImageVisual


class SceneManager:
    """Owns one pygfx ``gfx.Scene`` and the registry of visuals attached to it.

    Dimensionality (``"2d"`` or ``"3d"``) is set at construction time and
    determines which node from each visual is added to the scene graph and
    how ``build_slice_requests()`` interprets the ``ReslicingRequest``.

    Parameters
    ----------
    scene_id : UUID
        Unique identifier for this scene.
    dim : str
        Dimensionality, either ``"2d"`` or ``"3d"``.
    config : SceneRenderConfig or None
        Initial render configuration.  A default ``SceneRenderConfig`` is
        created if ``None`` is supplied.
    """

    def __init__(
        self,
        scene_id: UUID,
        dim: str,
        config: SceneRenderConfig | None = None,
    ) -> None:
        if dim not in ("2d", "3d"):
            raise ValueError(f"dim must be '2d' or '3d', got {dim!r}")
        self._scene_id = scene_id
        self._dim = dim
        self._scene = gfx.Scene()
        self._visuals: dict[UUID, GFXMultiscaleImageVisual] = {}
        self._config = config if config is not None else SceneRenderConfig()

    @property
    def scene_id(self) -> UUID:
        """Unique identifier for this scene."""
        return self._scene_id

    @property
    def dim(self) -> str:
        """Dimensionality of this scene (``"2d"`` or ``"3d"``)."""
        return self._dim

    @property
    def scene(self) -> gfx.Scene:
        """The pygfx Scene object. Passed to CanvasView via get_scene_fn."""
        return self._scene

    @property
    def config(self) -> SceneRenderConfig:
        """Live render configuration for this scene.

        The application may mutate fields on the returned object at any
        time.  The next reslicing cycle picks up the current values.
        """
        return self._config

    @property
    def visual_ids(self) -> list[UUID]:
        """IDs of all registered visuals."""
        return list(self._visuals.keys())

    def add_visual(self, visual: GFXMultiscaleImageVisual) -> None:
        """Register a visual and add its node to the scene graph.

        Adds ``node_3d`` if ``dim == "3d"``, ``node_2d`` if ``dim == "2d"``.

        Parameters
        ----------
        visual : GFXMultiscaleImageVisual
            The visual to register.

        Raises
        ------
        ValueError
            If the visual does not have a node for this dimensionality.
        """
        if self._dim == "3d":
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
        if self._dim == "3d" and visual.node_3d is not None:
            self._scene.remove(visual.node_3d)
        elif self._dim == "2d" and visual.node_2d is not None:
            self._scene.remove(visual.node_2d)

    def get_visual(self, visual_id: UUID) -> GFXMultiscaleImageVisual:
        """Return the registered visual for ``visual_id``.

        Parameters
        ----------
        visual_id : UUID
            ID of the visual to retrieve.

        Returns
        -------
        GFXMultiscaleImageVisual

        Raises
        ------
        KeyError
            If ``visual_id`` is not registered in this scene.
        """
        return self._visuals[visual_id]

    def build_slice_requests(
        self, request: ReslicingRequest
    ) -> dict[UUID, list[ChunkRequest]]:
        """Collect ChunkRequests from all (or targeted) registered visuals.

        Frustum planes, LOD thresholds, and force-level are derived from
        ``self.config`` at call time so application changes take effect on
        the next reslicing cycle without any re-registration.

        Parameters
        ----------
        request : ReslicingRequest
            The reslicing request.  If ``request.target_visual_ids`` is not
            ``None``, only visuals whose ID is in that set are processed.

        Returns
        -------
        dict[UUID, list[ChunkRequest]]
            Mapping of ``visual_model_id`` to that visual's ChunkRequests,
            ordered nearest-first within each list.  Visuals with no missing
            bricks (all cached) are omitted from the dict.
        """
        frustum_planes = (
            frustum_planes_from_corners(request.frustum_corners)
            if self._config.frustum_cull
            else None
        )

        result: dict[UUID, list[ChunkRequest]] = {}
        for visual_id, visual in self._visuals.items():
            if (
                request.target_visual_ids is not None
                and visual_id not in request.target_visual_ids
            ):
                continue

            n_levels = visual._volume_geometry.n_levels
            thresholds = self._compute_thresholds(request, n_levels)

            chunk_requests = visual.build_slice_request(
                camera_pos=request.camera_pos,
                frustum_planes=frustum_planes,
                thresholds=thresholds,
                force_level=self._config.force_level,
            )
            if chunk_requests:
                result[visual_id] = chunk_requests

        return result

    def _compute_thresholds(
        self, request: ReslicingRequest, n_levels: int
    ) -> list[float]:
        """Compute LOD distance thresholds from camera parameters and config.

        Parameters
        ----------
        request : ReslicingRequest
            The reslicing request providing FOV and screen height.
        n_levels : int
            Number of LOD levels for this visual.

        Returns
        -------
        list[float]
            One threshold per level boundary (length ``n_levels - 1``).
            Distance in world units at which the renderer transitions from
            level k to level k+1, scaled by ``self.config.lod_bias``.
        """
        focal_half_height = (request.screen_height_px / 2.0) / np.tan(
            request.fov_y_rad / 2.0
        )
        return [
            (2 ** (k - 1)) * focal_half_height * self._config.lod_bias
            for k in range(1, n_levels)
        ]
