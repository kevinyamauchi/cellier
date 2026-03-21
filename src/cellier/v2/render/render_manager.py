"""RenderManager — single top-level render-layer object."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cellier.v2.render.canvas_view import CanvasView
from cellier.v2.render.scene_manager import SceneManager
from cellier.v2.render.slice_coordinator import SliceCoordinator
from cellier.v2.slicer import AsyncSlicer

if TYPE_CHECKING:
    from uuid import UUID

    import pygfx as gfx
    from PySide6.QtWidgets import QWidget

    from cellier.v2.data.image import MultiscaleZarrDataStore
    from cellier.v2.render._requests import DimsState
    from cellier.v2.render._scene_config import SceneRenderConfig
    from cellier.v2.render.visuals._image import GFXMultiscaleImageVisual


class RenderManager:
    """Single top-level render-layer object.

    Owns the scene registry, canvas registry, shared async slicer, and
    slice coordinator.  Exposes three reslicing entry points that cover
    the common triggers: all scenes, one scene, or one visual.

    Construction is parameter-free; scenes, canvases, and visuals are
    registered via the ``add_*`` methods.
    """

    def __init__(self) -> None:
        self._scenes: dict[UUID, SceneManager] = {}
        self._canvases: dict[UUID, CanvasView] = {}
        self._canvas_to_scene: dict[UUID, UUID] = {}
        self._visual_to_scene: dict[UUID, UUID] = {}
        self._data_stores: dict[UUID, MultiscaleZarrDataStore] = {}
        self._slicer = AsyncSlicer(batch_size=8)
        self._slice_coordinator = SliceCoordinator(
            scenes=self._scenes,
            slicer=self._slicer,
            data_stores=self._data_stores,
        )

    def add_scene(self, scene_id: UUID, dim: str) -> SceneManager:
        """Create and register a new scene.

        Parameters
        ----------
        scene_id : UUID
            Unique identifier for the scene.
        dim : str
            Dimensionality, either ``"2d"`` or ``"3d"``.

        Returns
        -------
        SceneManager
            The newly created scene manager.
        """
        scene_manager = SceneManager(scene_id=scene_id, dim=dim)
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
            (e.g. ``fov``, ``depth_range``).

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
        self._canvases[canvas_id] = canvas_view
        self._canvas_to_scene[canvas_id] = scene_id
        return canvas_view

    def add_visual(
        self,
        scene_id: UUID,
        visual: GFXMultiscaleImageVisual,
        data_store: MultiscaleZarrDataStore,
    ) -> None:
        """Register a visual with a scene and its associated data store.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to add the visual to.
        visual : GFXMultiscaleImageVisual
            The render-layer visual object.
        data_store : MultiscaleZarrDataStore
            The data store that will serve chunk data for this visual.
        """
        self._scenes[scene_id].add_visual(visual)
        self._visual_to_scene[visual.visual_model_id] = scene_id
        self._data_stores[visual.visual_model_id] = data_store

    def get_scene_config(self, scene_id: UUID) -> SceneRenderConfig:
        """Return the live render config for ``scene_id``.

        The returned object is mutable.  Mutate fields in-place before
        calling any reslicing entry point and the new values will be used
        immediately.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene whose config should be returned.

        Returns
        -------
        SceneRenderConfig
        """
        return self._scenes[scene_id].config

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

    def trigger_update(self, dims_state: DimsState) -> None:
        """Reslice all visuals across all scenes.

        Builds one ``ReslicingRequest`` per canvas using the current camera
        state, then submits each to the ``SliceCoordinator``.  This is the
        method called by an Update button or similar explicit trigger.

        Parameters
        ----------
        dims_state : DimsState
            Current dimension display state.
        """
        for canvas_id, _scene_id in self._canvas_to_scene.items():
            canvas = self._canvases[canvas_id]
            request = canvas.capture_reslicing_request(dims_state)
            self._slice_coordinator.submit(request)

    def reslice_scene(self, scene_id: UUID, dims_state: DimsState) -> None:
        """Reslice all visuals in one scene.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to reslice.
        dims_state : DimsState
            Current dimension display state.
        """
        canvas = self._find_canvas_for_scene(scene_id)
        if canvas is None:
            return
        request = canvas.capture_reslicing_request(dims_state)
        self._slice_coordinator.submit(request)

    def reslice_visual(self, visual_id: UUID, dims_state: DimsState) -> None:
        """Reslice one visual.

        Looks up which scene owns ``visual_id``, builds a
        ``ReslicingRequest`` with ``target_visual_ids={visual_id}``, and
        submits it.  The camera snapshot still uses the full current camera
        state — LOD thresholds and frustum culling depend on it.

        Parameters
        ----------
        visual_id : UUID
            ID of the visual to reslice.
        dims_state : DimsState
            Current dimension display state.
        """
        scene_id = self._visual_to_scene[visual_id]
        canvas = self._find_canvas_for_scene(scene_id)
        if canvas is None:
            return
        request = canvas.capture_reslicing_request(
            dims_state, target_visual_ids=frozenset({visual_id})
        )
        self._slice_coordinator.submit(request)

    def _find_canvas_for_scene(self, scene_id: UUID) -> CanvasView | None:
        for canvas_id, s_id in self._canvas_to_scene.items():
            if s_id == scene_id:
                return self._canvases[canvas_id]
        return None
