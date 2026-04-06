"""RenderManager — single top-level render-layer object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cellier.v2.render._config import RenderManagerConfig
from cellier.v2.render._scene_config import VisualRenderConfig
from cellier.v2.render._temporal_accumulation import TemporalAccumulationPass
from cellier.v2.render.canvas_view import CanvasView
from cellier.v2.render.scene_manager import SceneManager
from cellier.v2.render.slice_coordinator import SliceCoordinator
from cellier.v2.slicer import AsyncSlicer

if TYPE_CHECKING:
    from uuid import UUID

    import pygfx as gfx
    from PySide6.QtWidgets import QWidget

    from cellier.v2.data._base_data_store import BaseDataStore
    from cellier.v2.render._requests import DimsState


class RenderManager:
    """Single top-level render-layer object.

    Owns the scene registry, canvas registry, shared async slicer, and
    slice coordinator.  Exposes three reslicing entry points that cover
    the common triggers: all scenes, one scene, or one visual.

    Construction is parameter-free; scenes, canvases, and visuals are
    registered via the ``add_*`` methods.
    """

    def __init__(self, config: RenderManagerConfig | None = None) -> None:
        if config is None:
            config = RenderManagerConfig()
        self._config = config
        self._scenes: dict[UUID, SceneManager] = {}
        self._canvases: dict[UUID, CanvasView] = {}
        self._canvas_to_scene: dict[UUID, UUID] = {}
        self._visual_to_scene: dict[UUID, UUID] = {}
        self._data_stores: dict[UUID, BaseDataStore] = {}
        self._slicer = AsyncSlicer(
            batch_size=config.slicing.batch_size,
            render_every=config.slicing.render_every,
        )
        self._slice_coordinator = SliceCoordinator(
            scenes=self._scenes,
            slicer=self._slicer,
            data_stores=self._data_stores,
        )
        self._temporal_pass = TemporalAccumulationPass(alpha=config.temporal.alpha)

    @property
    def config(self) -> RenderManagerConfig:
        """Current rendering performance configuration.

        Reflects live state: mutations via ``temporal_alpha`` and
        ``temporal_enabled`` setters are visible here immediately.
        """
        return self._config

    @property
    def temporal_alpha(self) -> float:
        """EMA floor weight for temporal accumulation."""
        return self._config.temporal.alpha

    @temporal_alpha.setter
    def temporal_alpha(self, value: float) -> None:
        self._config.temporal.alpha = value
        self._temporal_pass.alpha = value

    @property
    def temporal_enabled(self) -> bool:
        """Whether the temporal accumulation pass is active."""
        return self._config.temporal.enabled

    @temporal_enabled.setter
    def temporal_enabled(self, value: bool) -> None:
        self._config.temporal.enabled = value
        if not value:
            self._temporal_pass.reset()

    def add_scene(self, scene_id: UUID) -> SceneManager:
        """Create and register a new scene.

        Parameters
        ----------
        scene_id : UUID
            Unique identifier for the scene.

        Returns
        -------
        SceneManager
            The newly created scene manager.
        """
        scene_manager = SceneManager(scene_id=scene_id)
        self._scenes[scene_id] = scene_manager
        return scene_manager

    def add_canvas(
        self,
        canvas_id: UUID,
        scene_id: UUID,
        parent: QWidget | None = None,
        **canvas_view_kwargs,
    ) -> CanvasView:
        """Create a ``CanvasView``, register it, and return it.

        The caller embeds ``canvas_view.widget`` in their Qt layout.

        Parameters
        ----------
        canvas_id : UUID
            Unique identifier for this canvas.
        scene_id : UUID
            ID of the scene this canvas should render.
        parent : QWidget or None
            Parent widget for the underlying ``QRenderWidget``.
        **canvas_view_kwargs
            Additional keyword arguments forwarded to ``CanvasView.__init__``
            (e.g. ``dim``, ``fov``, ``depth_range``).

        Returns
        -------
        CanvasView
            The newly created canvas view.
        """
        canvas_view = CanvasView(
            canvas_id=canvas_id,
            scene_id=scene_id,
            get_scene_fn=self.get_scene,
            parent=parent,
            **canvas_view_kwargs,
        )
        # Wire up per-frame tick for visuals (e.g. jitter seed advance).
        canvas_view._tick_visuals_fn = self._make_tick_fn(scene_id)
        self._canvases[canvas_id] = canvas_view
        self._canvas_to_scene[canvas_id] = scene_id
        return canvas_view

    def add_visual(
        self,
        scene_id: UUID,
        visual: Any,
        data_store: BaseDataStore,
        displayed_axes: tuple[int, ...],
    ) -> None:
        """Register a visual with a scene and its associated data store.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to add the visual to.
        visual : GFXMultiscaleImageVisual | GFXImageMemoryVisual
            The render-layer visual object.
        data_store : BaseDataStore
            The data store that will serve chunk data for this visual.
        displayed_axes : tuple[int, ...]
            Current displayed axes from the scene's dims selection.  Passed to
            ``SceneManager.add_visual`` to select the initial node.
        """
        self._scenes[scene_id].add_visual(visual, displayed_axes)
        self._visual_to_scene[visual.visual_model_id] = scene_id
        self._data_stores[visual.visual_model_id] = data_store

    def _make_tick_fn(self, scene_id: UUID):
        """Return a callable that ticks all visuals in *scene_id*."""

        def _tick():
            sm = self._scenes.get(scene_id)
            if sm is None:
                return
            for vid in sm.visual_ids:
                vis = sm.get_visual(vid)
                if hasattr(vis, "tick"):
                    vis.tick()

        return _tick

    def get_scene(self, scene_id: UUID) -> gfx.Scene:
        """Return the pygfx Scene for ``scene_id``.

        Passed as a callable to ``CanvasView`` at construction time so
        ``CanvasView`` can render the scene without holding a direct
        ``SceneManager`` reference.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to retrieve.

        Returns
        -------
        gfx.Scene
        """
        return self._scenes[scene_id].scene

    def trigger_update(
        self,
        dims_state: DimsState,
        visual_configs: dict[UUID, VisualRenderConfig] | None = None,
    ) -> None:
        """Reslice all visuals across all scenes.

        Builds one ``ReslicingRequest`` per canvas using the current camera
        state, then submits each to the ``SliceCoordinator``.

        Parameters
        ----------
        dims_state : DimsState
            Current dimension display state.
        visual_configs : dict[UUID, VisualRenderConfig] or None
            Per-visual render configuration.  ``None`` falls back to defaults
            for all visuals.
        """
        if visual_configs is None:
            visual_configs = {}
        for canvas_id, _scene_id in self._canvas_to_scene.items():
            canvas = self._canvases[canvas_id]
            request = canvas.capture_reslicing_request(dims_state)
            self._slice_coordinator.submit(request, visual_configs)

    def reslice_scene(
        self,
        scene_id: UUID,
        dims_state: DimsState,
        visual_configs: dict[UUID, VisualRenderConfig] | None = None,
        target_visual_ids: frozenset[UUID] | None = None,
    ) -> None:
        """Reslice all visuals in one scene.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to reslice.
        dims_state : DimsState
            Current dimension display state.
        visual_configs : dict[UUID, VisualRenderConfig] or None
            Per-visual render configuration.  ``None`` falls back to defaults.
        target_visual_ids : frozenset[UUID] or None
            ``None`` reslices all visuals in the scene.
        """
        if visual_configs is None:
            visual_configs = {}
        canvas = self._find_canvas_for_scene(scene_id)
        if canvas is None:
            return
        request = canvas.capture_reslicing_request(
            dims_state, target_visual_ids=target_visual_ids
        )
        self._slice_coordinator.submit(request, visual_configs)

    def reslice_visual(
        self,
        visual_id: UUID,
        dims_state: DimsState,
        visual_config: VisualRenderConfig | None = None,
    ) -> None:
        """Reslice one visual.

        Looks up which scene owns ``visual_id``, builds a
        ``ReslicingRequest`` with ``target_visual_ids={visual_id}``, and
        submits it.

        Parameters
        ----------
        visual_id : UUID
            ID of the visual to reslice.
        dims_state : DimsState
            Current dimension display state.
        visual_config : VisualRenderConfig or None
            Render configuration for this visual.  ``None`` uses defaults.
        """
        cfg = visual_config if visual_config is not None else VisualRenderConfig()
        scene_id = self._visual_to_scene[visual_id]
        canvas = self._find_canvas_for_scene(scene_id)
        if canvas is None:
            return
        request = canvas.capture_reslicing_request(
            dims_state, target_visual_ids=frozenset({visual_id})
        )
        self._slice_coordinator.submit(request, {visual_id: cfg})

    def _find_canvas_for_scene(self, scene_id: UUID) -> CanvasView | None:
        for canvas_id, s_id in self._canvas_to_scene.items():
            if s_id == scene_id:
                return self._canvases[canvas_id]
        return None
