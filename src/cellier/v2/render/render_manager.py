"""RenderManager — single top-level render-layer object."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cellier.v2.events import DimsChangedEvent, EventBus
from cellier.v2.render._config import RenderManagerConfig
from cellier.v2.render._scene_config import VisualRenderConfig
from cellier.v2.render.canvas_view import CanvasView
from cellier.v2.render.scene_manager import SceneManager
from cellier.v2.render.slice_coordinator import SliceCoordinator
from cellier.v2.render.visuals._canvas_overlay import GFXCenteredAxes2D
from cellier.v2.slicer import AsyncSlicer
from cellier.v2.visuals._canvas_overlay import CenteredAxes2D

if TYPE_CHECKING:
    from uuid import UUID

    import pygfx as gfx
    from PySide6.QtWidgets import QWidget

    from cellier.v2.data._base_data_store import BaseDataStore
    from cellier.v2.render._requests import DimsState
    from cellier.v2.render.visuals._canvas_overlay import GFXCanvasOverlay
    from cellier.v2.render.visuals._image import GFXMultiscaleImageVisual
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual
    from cellier.v2.render.visuals._lines_memory import GFXLinesMemoryVisual
    from cellier.v2.render.visuals._mesh_memory import GFXMeshMemoryVisual
    from cellier.v2.render.visuals._points_memory import GFXPointsMemoryVisual
    from cellier.v2.visuals._canvas_overlay import CanvasOverlay

    _GFXVisual = (
        GFXMultiscaleImageVisual
        | GFXImageMemoryVisual
        | GFXPointsMemoryVisual
        | GFXLinesMemoryVisual
        | GFXMeshMemoryVisual
    )


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

    def connect_event_bus(self, event_bus: EventBus) -> None:
        """Subscribe internal components to *event_bus*.

        Must be called before the caller registers its own DimsChangedEvent
        handler so the SliceCoordinator invalidates stale 2D caches first.
        """
        event_bus.subscribe(
            DimsChangedEvent,
            self._slice_coordinator._on_dims_changed,
            owner_id=self._slice_coordinator.id,
        )

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
        for canvas in self._canvases.values():
            canvas._accum_pass.alpha = value

    @property
    def temporal_enabled(self) -> bool:
        """Whether the temporal accumulation pass is active."""
        return self._config.temporal.enabled

    @temporal_enabled.setter
    def temporal_enabled(self, value: bool) -> None:
        self._config.temporal.enabled = value
        for canvas in self._canvases.values():
            canvas._accum_pass.enabled = value

    def add_scene(self, scene_id: UUID, lighting: str = "none") -> SceneManager:
        """Create and register a new scene.

        Parameters
        ----------
        scene_id : UUID
            Unique identifier for the scene.
        lighting : str
            ``"none"`` (default) or ``"default"``.  Pass ``"default"`` to
            add ambient and directional lights — required for
            ``MeshPhongAppearance``.

        Returns
        -------
        SceneManager
            The newly created scene manager.
        """
        scene_manager = SceneManager(scene_id=scene_id, lighting=lighting)
        self._scenes[scene_id] = scene_manager
        return scene_manager

    def scene_has_lighting(self, scene_id: UUID) -> bool:
        """Return True if *scene_id* was created with lighting enabled."""
        return self._scenes[scene_id].has_lighting

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
        # Apply temporal config to the canvas's accumulation pass.
        canvas_view._accum_pass.alpha = self._config.temporal.alpha
        if not self._config.temporal.enabled:
            canvas_view._accum_pass.enabled = False
        # Wire up per-frame tick for visuals (e.g. jitter seed advance).
        canvas_view._tick_visuals_fn = self._make_tick_fn(scene_id)
        self._canvases[canvas_id] = canvas_view
        self._canvas_to_scene[canvas_id] = scene_id
        return canvas_view

    def add_visual(
        self,
        scene_id: UUID,
        visual: _GFXVisual,
        data_store: BaseDataStore,
        displayed_axes: tuple[int, ...],
    ) -> None:
        """Register a visual with a scene and its associated data store.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to add the visual to.
        visual : _GFXVisual
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

    def add_canvas_overlay(
        self,
        canvas_id: UUID,
        overlay_model: CanvasOverlay,
    ) -> None:
        """Build a GFX overlay and attach it to *canvas_id*.

        The GFX overlay is constructed from *overlay_model* and wired to the
        main camera of the canvas so that direction vectors are computed
        correctly.

        Parameters
        ----------
        canvas_id : UUID
            ID of the canvas that should receive the overlay.
        overlay_model : CanvasOverlay
            Model-layer description of the overlay.

        Raises
        ------
        KeyError
            If *canvas_id* is not registered.
        ValueError
            If *overlay_model* has an unrecognised type.
        """
        canvas = self._canvases[canvas_id]
        gfx_overlay = self._build_gfx_overlay(overlay_model, canvas)
        canvas.add_overlay(gfx_overlay)

    @staticmethod
    def _build_gfx_overlay(
        overlay_model: CanvasOverlay,
        canvas_view: CanvasView,
    ) -> GFXCanvasOverlay:
        """Construct the render-layer overlay for *overlay_model*.

        Parameters
        ----------
        overlay_model : CanvasOverlay
            Model-layer overlay description.  Dispatched on ``overlay_type``.
        canvas_view : CanvasView
            The canvas view that will own the overlay.  Provides the main
            camera reference.

        Returns
        -------
        GFXCanvasOverlay

        Raises
        ------
        ValueError
            If ``overlay_model.overlay_type`` is not recognised.
        """
        if isinstance(overlay_model, CenteredAxes2D):
            return GFXCenteredAxes2D(
                model=overlay_model,
                camera=canvas_view._camera,
            )
        raise ValueError(
            f"Unrecognised overlay_type: {overlay_model!r}.  "
            "Register the type in RenderManager._build_gfx_overlay."
        )

    def remove_visual(self, visual_id: UUID) -> None:
        """Remove a visual from its scene and deregister it.

        Parameters
        ----------
        visual_id : UUID
            ID of the visual to remove.
        """
        scene_id = self._visual_to_scene.pop(visual_id)
        self._data_stores.pop(visual_id)
        self._scenes[scene_id].remove_visual(visual_id)

    def remove_scene(self, scene_id: UUID) -> None:
        """Remove a scene and all its visuals and canvases.

        Drops all Python references so GC can reclaim GPU resources.
        pygfx has no explicit destroy API; reference dropping is sufficient.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to remove.
        """
        scene_manager = self._scenes.pop(scene_id)
        for vid in scene_manager.visual_ids:
            self._visual_to_scene.pop(vid, None)
            self._data_stores.pop(vid, None)
        # scene_manager goes out of scope here; GC drops gfx.Scene + all nodes.

        canvas_ids = [
            cid for cid, sid in self._canvas_to_scene.items() if sid == scene_id
        ]
        for cid in canvas_ids:
            self._canvas_to_scene.pop(cid)
            self._canvases.pop(cid)
        # CanvasView references dropped; GC handles Qt widget + wgpu renderer.

    def _make_tick_fn(self, scene_id: UUID):
        """Return a callable that ticks all visuals in *scene_id*."""

        def _tick():
            sm = self._scenes.get(scene_id)
            if sm is None:
                return
            for vid in sm.visual_ids:
                vis = sm.get_visual(vid)
                vis.tick()

        return _tick

    def get_scene(self, scene_id: UUID) -> gfx.Scene:
        """Return the pygfx Scene for ``scene_id``.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to retrieve.

        Returns
        -------
        gfx.Scene
        """
        return self._scenes[scene_id].scene

    def reslice_scene(
        self,
        scene_id: UUID,
        dims_state: DimsState,
        visual_configs: dict[UUID, VisualRenderConfig] | None = None,
        target_visual_ids: frozenset[UUID] | None = None,
    ) -> None:
        """Reslice all visuals in one scene.

        One reslicing request is submitted per registered canvas so that each
        canvas uses its own camera state for LOD and frustum-culling decisions.

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
        canvases = self._find_canvases_for_scene(scene_id)
        for canvas in canvases:
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

        Looks up which scene owns ``visual_id``, then submits one
        ``ReslicingRequest`` per registered canvas so that each canvas uses
        its own camera state.

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
        canvases = self._find_canvases_for_scene(scene_id)
        for canvas in canvases:
            request = canvas.capture_reslicing_request(
                dims_state, target_visual_ids=frozenset({visual_id})
            )
            self._slice_coordinator.submit(request, {visual_id: cfg})

    def look_at_visual(
        self,
        visual_id: UUID,
        canvas_id: UUID,
        view_direction: tuple[float, float, float] = (-1, -1, -1),
        up: tuple[float, float, float] = (0, 0, 1),
    ) -> None:
        """Fit a canvas camera to a visual's bounding box.

        Parameters
        ----------
        visual_id : UUID
            ID of the target visual.
        canvas_id : UUID
            ID of the canvas whose camera should be fitted.
        view_direction : tuple[float, float, float]
            Camera look direction vector (need not be normalized).
        up : tuple[float, float, float]
            Camera up vector.
        """
        scene_id = self._visual_to_scene[visual_id]
        gfx_scene = self.get_scene(scene_id)
        self._canvases[canvas_id]._camera.show_object(
            gfx_scene, view_dir=view_direction, up=up
        )

    def set_camera_depth_range(
        self,
        canvas_id: UUID,
        depth_range: tuple[float, float],
    ) -> None:
        """Set the near/far clip distances for a canvas camera.

        Parameters
        ----------
        canvas_id : UUID
            ID of the target canvas.
        depth_range : tuple[float, float]
            ``(near, far)`` clip distances in world units.
        """
        self._canvases[canvas_id].set_depth_range(depth_range)

    def _find_canvases_for_scene(self, scene_id: UUID) -> list[CanvasView]:
        """Return all canvases registered for *scene_id*.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to look up.

        Returns
        -------
        list[CanvasView]
            All canvas views rendering the scene.  Empty if none registered.
        """
        return [
            self._canvases[cid]
            for cid, sid in self._canvas_to_scene.items()
            if sid == scene_id
        ]
