"""CellierController — single entry-point for building Cellier v2 viewers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable, overload
from uuid import UUID, uuid4

import numpy as np

from cellier.v2.data.image._image_memory_store import ImageMemoryStore
from cellier.v2.events import (
    AppearanceChangedEvent,
    CameraChangedEvent,
    DimsChangedEvent,
    EventBus,
    SceneAddedEvent,
    SubscriptionHandle,
    TransformChangedEvent,
    VisualAddedEvent,
    VisualVisibilityChangedEvent,
)
from cellier.v2.logging import _CAMERA_LOGGER
from cellier.v2.render._scene_config import VisualRenderConfig
from cellier.v2.render.render_manager import RenderManager
from cellier.v2.render.visuals._image import GFXMultiscaleImageVisual
from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual
from cellier.v2.scene.cameras import (
    OrbitCameraController,
    OrthographicCamera,
    PerspectiveCamera,
)
from cellier.v2.scene.canvas import Canvas
from cellier.v2.scene.dims import (
    AxisAlignedSelection,
    CoordinateSystem,
    DimsManager,
)
from cellier.v2.scene.scene import Scene
from cellier.v2.viewer_model import DataManager, ViewerModel
from cellier.v2.visuals._image import MultiscaleImageVisual
from cellier.v2.visuals._image_memory import ImageMemoryAppearance, ImageVisual

if TYPE_CHECKING:
    import pathlib

    from psygnal import EmissionInfo
    from PySide6.QtWidgets import QWidget

    from cellier.v2._state import CameraState, DimsState
    from cellier.v2.data._base_data_store import BaseDataStore
    from cellier.v2.transform import AffineTransform
    from cellier.v2.visuals._image import ImageAppearance


# Appearance fields that require a reslice (not just a GPU material update).
_RESLICE_FIELDS: frozenset[str] = frozenset({"lod_bias", "force_level", "frustum_cull"})

DEFAULT_CAMERA_SETTLE_THRESHOLD_S: float = 0.3


class CellierController:
    """The main class for constructing and controlling a cellier visualization.

    Wraps a ViewerModel (model layer) and a RenderManager (render layer).
    """

    def __init__(
        self,
        widget_parent: QWidget | None = None,
        slicer_batch_size: int = 8,
        slicer_render_every: int = 1,
        camera_settle_threshold_s: float = DEFAULT_CAMERA_SETTLE_THRESHOLD_S,
        camera_reslice_enabled: bool = True,
    ) -> None:
        self._widget_parent = widget_parent
        self._model = ViewerModel(data=DataManager())
        self._render_manager = RenderManager(
            slicer_batch_size=slicer_batch_size,
            slicer_render_every=slicer_render_every,
        )
        # Reverse map: visual_model_id → scene_id (model layer mirror of render layer)
        self._visual_to_scene: dict[UUID, UUID] = {}
        # Reverse map: canvas_id → scene_id
        self._canvas_to_scene: dict[UUID, UUID] = {}
        # Forward map: scene_id → list[canvas_id]
        self._scene_to_canvases: dict[UUID, list[UUID]] = {}
        # Event bus
        self._id: UUID = uuid4()
        self._event_bus: EventBus = EventBus()
        # Cache of last-known displayed_axes per scene for change detection
        self._dims_cache: dict[UUID, tuple[int, ...]] = {}
        # Handles for externally-registered callbacks
        self._external_handles: list[SubscriptionHandle] = []
        # Camera settle
        self._camera_settle_threshold_s: float = camera_settle_threshold_s
        self._camera_reslice_enabled: bool = camera_reslice_enabled
        self._settle_tasks: dict[UUID, asyncio.Task] = {}
        self._event_bus.subscribe(
            CameraChangedEvent,
            self._on_camera_changed,
            owner_id=self._id,
        )

    def set_widget_parent(self, parent: QWidget) -> None:
        """Set the Qt parent for subsequently created canvas widgets."""
        self._widget_parent = parent

    # ------------------------------------------------------------------
    # Construction class methods (stubs)
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: ViewerModel,
        widget_parent: QWidget | None = None,
    ) -> CellierController:
        """Construct a controller from a pre-built ViewerModel.

        Not implemented in Phase 1.  Materializing the full render layer from
        an existing model requires reading level_shapes from every data store
        and is deferred until needed.
        """
        raise NotImplementedError(
            "CellierController.from_model is not implemented in Phase 1."
        )

    @classmethod
    def from_file(
        cls,
        path: str | pathlib.Path,
        widget_parent: QWidget | None = None,
    ) -> CellierController:
        """Deserialize a ViewerModel from disk and construct a controller.

        Not implemented in Phase 1.
        """
        raise NotImplementedError(
            "CellierController.from_file is not implemented in Phase 1."
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_file(self, path: str | pathlib.Path) -> None:
        """Serialize the current model state to a JSON file.

        Camera positions in the model are NOT synchronized from the render
        layer before saving in Phase 1 — the model cameras remain at their
        construction-time defaults.
        """
        self._model.to_file(path)

    def to_model(self) -> ViewerModel:
        """Return a copy of the current model state.

        Camera positions are NOT synchronized from the render layer in
        Phase 1 (same caveat as to_file).
        """
        return self._model.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Scene management
    # ------------------------------------------------------------------

    def add_scene(
        self,
        dim: str,
        coordinate_system: CoordinateSystem,
        name: str,
    ) -> Scene:
        """Create and register a new empty scene.

        Parameters
        ----------
        dim : str
            ``"2d"`` or ``"3d"``.
        coordinate_system : CoordinateSystem
            World coordinate system for the scene.
        name : str
            Human-readable name.

        Returns
        -------
        Scene
            The live model object.  Use ``scene.id`` in subsequent calls.
        """
        ndim = len(coordinate_system.axis_labels)
        if dim == "3d":
            displayed_axes = tuple(range(ndim))
            slice_indices_dict: dict[int, int] = {}
        elif dim == "2d":
            # Display the last two axes; slice all others at 0.
            displayed_axes = tuple(range(ndim - 2, ndim))
            slice_indices_dict = {i: 0 for i in range(ndim - 2)}
        else:
            raise ValueError(f"dim must be '2d' or '3d', got {dim!r}")

        dims = DimsManager(
            coordinate_system=coordinate_system,
            selection=AxisAlignedSelection(
                displayed_axes=displayed_axes,
                slice_indices=slice_indices_dict,
            ),
        )
        scene = Scene(name=name, dims=dims)
        self._model.scenes[scene.id] = scene
        self._render_manager.add_scene(scene.id, dim=dim)
        self._scene_to_canvases[scene.id] = []
        self._wire_dims_model(scene)
        self._event_bus.emit(SceneAddedEvent(source_id=self._id, scene_id=scene.id))
        return scene

    # ------------------------------------------------------------------
    # Data store management
    # ------------------------------------------------------------------

    def add_data_store(self, data_store: BaseDataStore) -> BaseDataStore:
        """Register a data store and return it.

        Parameters
        ----------
        data_store : BaseDataStore
            The store to register.

        Returns
        -------
        BaseDataStore
            The same object passed in.
        """
        self._model.data.stores[data_store.id] = data_store
        return data_store

    # ------------------------------------------------------------------
    # Visual management
    # ------------------------------------------------------------------

    @overload
    def add_image(
        self,
        data: ImageMemoryStore,
        scene_id: UUID,
        appearance: ImageMemoryAppearance,
        name: str = ...,
    ) -> ImageVisual: ...

    @overload
    def add_image(
        self,
        data: BaseDataStore,
        scene_id: UUID,
        appearance: ImageAppearance,
        name: str = ...,
        block_size: int = ...,
        gpu_budget_bytes: int = ...,
        threshold: float | None = ...,
        interpolation: str = ...,
        use_brick_shader: bool = ...,
        voxel_spacing: ... = ...,
    ) -> MultiscaleImageVisual: ...

    def add_image(
        self,
        data,
        scene_id: UUID,
        appearance,
        name: str = "image",
        block_size: int = 32,
        gpu_budget_bytes: int = 1 * 1024**3,
        threshold: float | None = None,
        interpolation: str = "linear",
        use_brick_shader: bool = False,
        voxel_spacing=None,
    ) -> MultiscaleImageVisual | ImageVisual:
        """Add an image visual to a scene.

        Dispatches to the appropriate rendering path based on the type of
        ``data``:

        - ``ImageMemoryStore`` → ``GFXImageMemoryVisual`` backed by
          ``gfx.Image`` (2D) or ``gfx.Volume`` (3D). No brick cache; the
          full slice is uploaded on every reslice.
        - ``MultiscaleZarrDataStore`` → ``GFXMultiscaleImageVisual`` backed
          by a brick cache + LUT indirection system. Supports LOD, frustum
          culling, and out-of-core streaming.

        Parameters
        ----------
        data : ImageMemoryStore | MultiscaleZarrDataStore
            The data source.
        scene_id : UUID
            ID of an existing scene (returned by ``add_scene``).
        appearance : ImageMemoryAppearance | ImageAppearance
            Appearance parameters. Must match the type of ``data``.
        name : str
            Human-readable label for the visual. Default ``"image"``.
        block_size : int
            Brick side length in voxels. Only used for multiscale path.
            Default 32.
        gpu_budget_bytes : int
            GPU memory budget for the brick cache. Only used for multiscale
            path. Default 1 GiB.
        threshold : float
            Isosurface threshold for 3D raycast rendering. Only used for
            multiscale path. Default 0.2.
        interpolation : str
            Sampler filter ``"linear"`` or ``"nearest"``. Only used for
            multiscale path. Default ``"linear"``.
        use_brick_shader: bool
            If True, use the experimental brick shader.
            Default is False.
        voxel_spacing: tuple[float, ...] | None
            The spacing of the voxels in world coordinates.

        Returns
        -------
        ImageVisual
            When ``data`` is an ``ImageMemoryStore``.
        MultiscaleImageVisual
            When ``data`` is a ``MultiscaleZarrDataStore``.
        """
        if isinstance(data, ImageMemoryStore):
            return self._add_image_memory(data, scene_id, appearance, name)
        else:
            return self._add_image_multiscale(
                data,
                scene_id,
                appearance,
                name,
                block_size,
                gpu_budget_bytes,
                threshold,
                interpolation,
                use_brick_shader=use_brick_shader,
                voxel_spacing=voxel_spacing,
            )

    def _add_image_memory(
        self,
        data: ImageMemoryStore,
        scene_id: UUID,
        appearance: ImageMemoryAppearance,
        name: str,
    ) -> ImageVisual:
        """Add an in-memory image visual to a scene."""
        # ── 1. Register the data store if needed ────────────────────────
        if data.id not in self._model.data.stores:
            self._model.data.stores[data.id] = data

        # ── 2. Determine render mode from scene dimensionality ──────────
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_mode = "3d" if len(displayed_axes) == 3 else "2d"

        # ── 3. Build model-layer objects ────────────────────────────────
        visual_model = ImageVisual(
            name=name,
            data_store_id=str(data.id),
            appearance=appearance,
        )
        scene.visuals.append(visual_model)

        # ── 4. Build render-layer object ────────────────────────────────
        gfx_visual = GFXImageMemoryVisual(
            visual_model=visual_model,
            data_store=data,
            render_mode=render_mode,
        )

        # ── 5. Register with RenderManager and controller maps ──────────
        self._render_manager.add_visual(scene_id, gfx_visual, data)
        self._visual_to_scene[visual_model.id] = scene_id

        # ── 6. Wire appearance/transform bridges and EventBus subscriptions
        self._wire_appearance(visual_model)
        self._wire_transform(visual_model, scene_id)
        self._event_bus.subscribe(
            AppearanceChangedEvent,
            gfx_visual.on_appearance_changed,
            entity_id=visual_model.id,
            owner_id=visual_model.id,
        )
        self._event_bus.subscribe(
            VisualVisibilityChangedEvent,
            gfx_visual.on_visibility_changed,
            entity_id=visual_model.id,
            owner_id=visual_model.id,
        )
        self._event_bus.subscribe(
            TransformChangedEvent,
            gfx_visual.on_transform_changed,
            entity_id=visual_model.id,
            owner_id=visual_model.id,
        )
        self._event_bus.emit(
            VisualAddedEvent(
                source_id=self._id,
                scene_id=scene_id,
                visual_id=visual_model.id,
            )
        )
        return visual_model

    def _add_image_multiscale(
        self,
        data: BaseDataStore,
        scene_id: UUID,
        appearance: ImageAppearance,
        name: str,
        block_size: int,
        gpu_budget_bytes: int,
        threshold: float | None,
        interpolation: str,
        use_brick_shader: bool = False,
        voxel_spacing=None,
    ) -> MultiscaleImageVisual:
        """Add a multiscale image visual to a scene."""
        if data.id not in self._model.data.stores:
            self._model.data.stores[data.id] = data

        # Write explicit threshold into appearance for backward compat.
        if threshold is not None:
            appearance.iso_threshold = threshold

        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes

        visual_model = MultiscaleImageVisual(
            name=name,
            data_store_id=str(data.id),
            level_transforms=data.level_transforms,
            appearance=appearance,
        )

        scene.visuals.append(visual_model)

        render_modes = {"3d"} if len(displayed_axes) == 3 else {"2d"}

        level_shapes = list(data.level_shapes)
        gfx_visual = GFXMultiscaleImageVisual.from_cellier_model(
            model=visual_model,
            level_shapes=level_shapes,
            displayed_axes=displayed_axes,
            render_modes=render_modes,
            block_size=block_size,
            gpu_budget_bytes=gpu_budget_bytes,
            interpolation=interpolation,
            use_brick_shader=use_brick_shader,
            voxel_spacing=voxel_spacing,
        )

        self._render_manager.add_visual(scene_id, gfx_visual, data)
        self._visual_to_scene[visual_model.id] = scene_id
        self._wire_appearance(visual_model)
        self._wire_transform(visual_model, scene_id)
        self._event_bus.subscribe(
            AppearanceChangedEvent,
            gfx_visual.on_appearance_changed,
            entity_id=visual_model.id,
            owner_id=visual_model.id,
        )
        self._event_bus.subscribe(
            VisualVisibilityChangedEvent,
            gfx_visual.on_visibility_changed,
            entity_id=visual_model.id,
            owner_id=visual_model.id,
        )
        self._event_bus.subscribe(
            TransformChangedEvent,
            gfx_visual.on_transform_changed,
            entity_id=visual_model.id,
            owner_id=visual_model.id,
        )
        self._event_bus.emit(
            VisualAddedEvent(
                source_id=self._id,
                scene_id=scene_id,
                visual_id=visual_model.id,
            )
        )
        return visual_model

    def add_visual(
        self,
        scene_id: UUID,
        visual_model: MultiscaleImageVisual,
        data_store_id: UUID,
        block_size: int = 32,
        gpu_budget_bytes: int = 1 * 1024**3,
        threshold: float = 0.2,
        interpolation: str = "linear",
    ) -> MultiscaleImageVisual:
        """Register a pre-built visual model with a scene.

        Use this when you have already constructed a MultiscaleImageVisual
        model yourself.  For the common case, prefer ``add_image()``.

        Parameters
        ----------
        scene_id : UUID
            ID of an existing scene.
        visual_model : MultiscaleImageVisual
            The pre-built model.
        data_store_id : UUID
            ID of a data store already registered via ``add_data_store()``.
        block_size : int
            Brick side length in voxels.  Default 32.
        gpu_budget_bytes : int
            GPU memory budget for the brick cache.  Default 1 GiB.
        threshold : float
            Isosurface threshold for 3D raycast rendering.  Default 0.2.
        interpolation : str
            Sampler filter ``"linear"`` or ``"nearest"``.  Default ``"linear"``.

        Returns
        -------
        MultiscaleImageVisual
            The same model passed in.
        """
        data_store = self._model.data.stores[data_store_id]
        scene = self._model.scenes[scene_id]
        scene.visuals.append(visual_model)

        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = {"3d"} if len(displayed_axes) == 3 else {"2d"}
        level_shapes = list(data_store.level_shapes)
        gfx_visual = GFXMultiscaleImageVisual.from_cellier_model(
            model=visual_model,
            level_shapes=level_shapes,
            displayed_axes=displayed_axes,
            render_modes=render_modes,
            block_size=block_size,
            gpu_budget_bytes=gpu_budget_bytes,
            threshold=threshold,
            interpolation=interpolation,
        )
        self._render_manager.add_visual(scene_id, gfx_visual, data_store)
        self._visual_to_scene[visual_model.id] = scene_id
        return visual_model

    # ------------------------------------------------------------------
    # Canvas management
    # ------------------------------------------------------------------

    def add_canvas(
        self,
        scene_id: UUID,
        available_dims: str | None = None,
        fov: float = 70.0,
        depth_range: tuple[float, float] = (1.0, 8000.0),
    ) -> QWidget:
        """Create a canvas attached to a scene and return its embeddable widget.

        Parameters
        ----------
        scene_id : UUID
            ID of an existing scene.
        available_dims : str or None
            ``"3d"`` or ``"2d"``.  If ``None``, inferred from the scene's
            dimensionality.
        fov : float
            Vertical field of view in degrees (perspective/3D only).
        depth_range : tuple[float, float]
            ``(near, far)`` clip distances.

        Returns
        -------
        QWidget
            The render widget.  Embed with ``layout.addWidget(widget)``.
        """
        canvas_id = uuid4()

        # Infer dimensionality from the scene if not specified.
        if available_dims is None:
            scene = self._model.scenes[scene_id]
            available_dims = (
                "3d" if len(scene.dims.selection.displayed_axes) == 3 else "2d"
            )

        camera_model = PerspectiveCamera(
            fov=fov,
            near_clipping_plane=depth_range[0],
            far_clipping_plane=depth_range[1],
            controller=OrbitCameraController(enabled=True),
        )
        canvas_model = Canvas(cameras={available_dims: camera_model})
        self._model.scenes[scene_id].canvases[canvas_model.id] = canvas_model

        canvas_view = self._render_manager.add_canvas(
            canvas_id,
            scene_id,
            parent=self._widget_parent,
            dim=available_dims,
            fov=fov,
            depth_range=depth_range,
        )
        canvas_view.set_event_bus(self._event_bus)

        self._canvas_to_scene[canvas_id] = scene_id
        self._scene_to_canvases[scene_id].append(canvas_id)

        gfx_scene = self._render_manager.get_scene(scene_id)
        canvas_view.show_object(gfx_scene)

        return canvas_view.widget

    # ------------------------------------------------------------------
    # Scene and visual lookup
    # ------------------------------------------------------------------

    def get_scene(self, scene_id: UUID) -> Scene:
        """Return the live Scene model for scene_id."""
        return self._model.scenes[scene_id]

    def get_scene_by_name(self, name: str) -> Scene:
        """Return the live Scene model for the given name.

        Raises KeyError if no scene with that name exists.
        """
        for scene in self._model.scenes.values():
            if scene.name == name:
                return scene
        raise KeyError(f"No scene named {name!r}")

    def get_visual_model(self, visual_id: UUID) -> MultiscaleImageVisual:
        """Return the live visual model for visual_id.

        Searches all scenes.  Raises KeyError if not found.
        """
        for scene in self._model.scenes.values():
            for visual in scene.visuals:
                if visual.id == visual_id:
                    return visual
        raise KeyError(f"No visual with id {visual_id}")

    # ------------------------------------------------------------------
    # Reslicing
    # ------------------------------------------------------------------

    def _build_visual_configs_for_scene(
        self, scene_id: UUID
    ) -> dict[UUID, VisualRenderConfig]:
        """Build a VisualRenderConfig dict from all visuals in a scene."""
        scene = self._model.scenes[scene_id]
        configs: dict[UUID, VisualRenderConfig] = {}
        for visual in scene.visuals:
            if isinstance(visual, MultiscaleImageVisual):
                configs[visual.id] = VisualRenderConfig(
                    lod_bias=visual.appearance.lod_bias,
                    force_level=visual.appearance.force_level,
                    frustum_cull=visual.appearance.frustum_cull,
                )
        return configs

    def _dims_state_for_scene(self, scene_id: UUID) -> DimsState:
        """Derive a DimsState from the scene's DimsManager."""
        return self._model.scenes[scene_id].dims.to_state()

    # ------------------------------------------------------------------
    # psygnal bridges
    # ------------------------------------------------------------------

    def _wire_dims_model(self, scene: Scene) -> None:
        """Subscribe to all field changes on a scene's DimsManager."""
        self._dims_cache[scene.id] = scene.dims.selection.displayed_axes
        scene.dims.events.connect(self._make_dims_handler(scene.id))

    def _make_dims_handler(self, scene_id: UUID) -> Callable:
        """Return a psygnal catch-all handler for a scene's DimsManager."""

        def _on_dims_psygnal(info: EmissionInfo) -> None:
            new_state = self._model.scenes[scene_id].dims.to_state()
            prev_axes = self._dims_cache[scene_id]
            displayed_axes_changed = prev_axes != new_state.selection.displayed_axes
            self._dims_cache[scene_id] = new_state.selection.displayed_axes
            if displayed_axes_changed:
                self._rebuild_visuals_geometry(
                    scene_id, new_state.selection.displayed_axes
                )
            self._event_bus.emit(
                DimsChangedEvent(
                    source_id=self._id,
                    scene_id=scene_id,
                    dims_state=new_state,
                    displayed_axes_changed=displayed_axes_changed,
                )
            )

        return _on_dims_psygnal

    def _rebuild_visuals_geometry(
        self, scene_id: UUID, displayed_axes: tuple[int, ...]
    ) -> None:
        """Rebuild geometry for all visuals in a scene after displayed_axes change."""
        scene = self._model.scenes[scene_id]
        scene_manager = self._render_manager._scenes[scene_id]

        for visual_model in scene.visuals:
            data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
            gfx_visual = scene_manager.get_visual(visual_model.id)

            # Only multiscale visuals support geometry rebuilding on
            # displayed_axes change; in-memory visuals are static.
            if not hasattr(gfx_visual, "rebuild_geometry"):
                continue

            level_shapes = list(data_store.level_shapes)
            old_node, new_node = gfx_visual.rebuild_geometry(
                level_shapes, displayed_axes
            )

            # Swap node in the pygfx scene graph.
            if old_node is not None:
                scene_manager.scene.remove(old_node)
            if new_node is not None:
                scene_manager.scene.add(new_node)

    def _wire_transform(
        self, visual: MultiscaleImageVisual | ImageVisual, scene_id: UUID
    ) -> None:
        """Subscribe to transform field changes on a visual model."""
        visual.events.transform.connect(
            self._make_transform_handler(visual.id, scene_id)
        )

    def _make_transform_handler(self, visual_id: UUID, scene_id: UUID) -> Callable:
        """Return a handler that emits TransformChangedEvent and triggers reslice."""

        def _on_transform(new_transform: AffineTransform) -> None:
            self._event_bus.emit(
                TransformChangedEvent(
                    source_id=self._id,
                    scene_id=scene_id,
                    visual_id=visual_id,
                    transform=new_transform,
                )
            )
            self.reslice_scene(scene_id)

        return _on_transform

    def _wire_appearance(self, visual: MultiscaleImageVisual | ImageVisual) -> None:
        """Subscribe to all field changes on a visual's appearance model."""
        visual.appearance.events.connect(self._make_appearance_handler(visual.id))

    def _make_appearance_handler(self, visual_id: UUID) -> Callable:
        """Return a psygnal catch-all handler for a visual's ImageAppearance."""

        def _on_appearance_psygnal(info: EmissionInfo) -> None:
            field_name: str = info.signal.name
            new_value = info.args[0]
            if field_name == "visible":
                self._event_bus.emit(
                    VisualVisibilityChangedEvent(
                        source_id=self._id,
                        visual_id=visual_id,
                        visible=new_value,
                    )
                )
            else:
                self._event_bus.emit(
                    AppearanceChangedEvent(
                        source_id=self._id,
                        visual_id=visual_id,
                        field_name=field_name,
                        new_value=new_value,
                        requires_reslice=(field_name in _RESLICE_FIELDS),
                    )
                )

        return _on_appearance_psygnal

    def reslice_all(self) -> None:
        """Trigger a data load for all visuals across all scenes."""
        for scene_id in self._model.scenes:
            self.reslice_scene(scene_id)

    def reslice_scene(self, scene_id: UUID) -> None:
        """Trigger a data load for all visuals in one scene."""
        dims_state = self._dims_state_for_scene(scene_id)
        visual_configs = self._build_visual_configs_for_scene(scene_id)
        self._render_manager.reslice_scene(scene_id, dims_state, visual_configs)

    def reslice_visual(self, visual_id: UUID) -> None:
        """Trigger a data load for one visual."""
        scene_id = self._visual_to_scene[visual_id]
        dims_state = self._dims_state_for_scene(scene_id)
        visual = self.get_visual_model(visual_id)
        if isinstance(visual, MultiscaleImageVisual):
            cfg = VisualRenderConfig(
                lod_bias=visual.appearance.lod_bias,
                force_level=visual.appearance.force_level,
                frustum_cull=visual.appearance.frustum_cull,
            )
        else:
            cfg = VisualRenderConfig()
        self._render_manager.reslice_visual(visual_id, dims_state, cfg)

    # ------------------------------------------------------------------
    # Camera settle
    # ------------------------------------------------------------------

    @property
    def camera_reslice_enabled(self) -> bool:
        """Whether camera movement triggers automatic reslicing."""
        return self._camera_reslice_enabled

    @camera_reslice_enabled.setter
    def camera_reslice_enabled(self, value: bool) -> None:
        self._camera_reslice_enabled = value
        if not value:
            for task in self._settle_tasks.values():
                if not task.done():
                    task.cancel()
            self._settle_tasks.clear()

    def _on_camera_changed(self, event: CameraChangedEvent) -> None:
        """Synchronous bus handler — updates camera model and schedules settle task."""
        self._update_camera_model(event.scene_id, event.camera_state)

        if not self._camera_reslice_enabled:
            return

        canvas_id = event.source_id
        existing = self._settle_tasks.get(canvas_id)
        if existing is not None and not existing.done():
            _CAMERA_LOGGER.debug(
                "settle_cancel  canvas=%s  scene=%s",
                canvas_id,
                event.scene_id,
            )
            existing.cancel()

        _CAMERA_LOGGER.debug(
            "settle_schedule  canvas=%s  scene=%s  threshold=%.3fs",
            canvas_id,
            event.scene_id,
            self._camera_settle_threshold_s,
        )
        self._settle_tasks[canvas_id] = asyncio.create_task(
            self._settle_after(canvas_id, event.scene_id)
        )

    def _update_camera_model(self, scene_id: UUID, camera_state: CameraState) -> None:
        """Write a CameraState snapshot back into the model-layer camera.

        Branches on the actual model type (not camera_state.camera_type)
        because add_canvas may create a PerspectiveCamera even for 2D scenes.
        """
        scene = self._model.scenes[scene_id]
        # Find the first canvas and its first camera.
        canvas_model = next(iter(scene.canvases.values()))
        camera_model = next(iter(canvas_model.cameras.values()))

        # Common fields present on both camera model types.
        camera_model.position = np.array(camera_state.position, dtype=np.float32)
        camera_model.rotation = np.array(camera_state.rotation, dtype=np.float32)
        camera_model.zoom = camera_state.zoom
        camera_model.near_clipping_plane = camera_state.depth_range[0]
        camera_model.far_clipping_plane = camera_state.depth_range[1]

        # Type-specific fields.
        if isinstance(camera_model, PerspectiveCamera):
            camera_model.up_direction = np.array(camera_state.up, dtype=np.float32)
            camera_model.fov = camera_state.fov
        elif isinstance(camera_model, OrthographicCamera):
            camera_model.width = camera_state.extent[0]
            camera_model.height = camera_state.extent[1]

    async def _settle_after(self, canvas_id: UUID, scene_id: UUID) -> None:
        """Wait for the settle threshold, then reslice camera-sensitive visuals."""
        try:
            await asyncio.sleep(self._camera_settle_threshold_s)
        except asyncio.CancelledError:
            raise

        scene = self._model.scenes[scene_id]
        target_ids = frozenset(v.id for v in scene.visuals if v.requires_camera_reslice)

        if not target_ids:
            return

        dims_state = self._dims_state_for_scene(scene_id)
        visual_configs = self._build_visual_configs_for_scene(scene_id)

        _CAMERA_LOGGER.info(
            "settle_reslice  scene=%s  visuals=%d",
            scene_id,
            len(target_ids),
        )

        self._render_manager.reslice_scene(
            scene_id=scene_id,
            dims_state=dims_state,
            visual_configs=visual_configs,
            target_visual_ids=target_ids,
        )

    # ------------------------------------------------------------------
    # Camera operations
    # ------------------------------------------------------------------

    def look_at_visual(
        self,
        visual_id: UUID,
        view_direction: tuple[float, float, float] = (-1, -1, -1),
        up: tuple[float, float, float] = (0, 0, 1),
    ) -> None:
        """Fit the camera to a visual's bounding box.

        Parameters
        ----------
        visual_id : UUID
            ID of the target visual.
        view_direction : tuple[float, float, float]
            Camera look direction vector (need not be normalized).
        up : tuple[float, float, float]
            Camera up vector.
        """
        scene_id = self._visual_to_scene[visual_id]
        canvas_ids = self._scene_to_canvases.get(scene_id, [])
        gfx_scene = self._render_manager.get_scene(scene_id)
        for canvas_id in canvas_ids:
            canvas_view = self._render_manager._canvases[canvas_id]
            canvas_view._camera.show_object(gfx_scene, view_dir=view_direction, up=up)

    def set_camera_depth_range(
        self,
        scene_id: UUID,
        depth_range: tuple[float, float],
    ) -> None:
        """Set the near/far clip distances for all canvases attached to a scene.

        Parameters
        ----------
        scene_id : UUID
            ID of the target scene.
        depth_range : tuple[float, float]
            ``(near, far)`` clip distances in world units.
        """
        canvas_ids = self._scene_to_canvases.get(scene_id, [])
        for canvas_id in canvas_ids:
            canvas_view = self._render_manager._canvases[canvas_id]
            canvas_view.set_depth_range(depth_range)

    # ------------------------------------------------------------------
    # Stubs for future features
    # ------------------------------------------------------------------

    def add_points(self, *args, **kwargs):
        """Not implemented in Phase 1."""
        raise NotImplementedError("add_points is not implemented in Phase 1.")

    def add_labels(self, *args, **kwargs):
        """Not implemented in Phase 1."""
        raise NotImplementedError("add_labels is not implemented in Phase 1.")

    def add_mesh(self, *args, **kwargs):
        """Not implemented in Phase 1."""
        raise NotImplementedError("add_mesh is not implemented in Phase 1.")

    def remove_scene(self, scene_id: UUID) -> None:
        """Clean up bus subscriptions for a scene and its canvases.

        Render-layer teardown is not yet implemented.
        """
        for canvas_id in self._scene_to_canvases.pop(scene_id, []):
            self._event_bus.unsubscribe_all(canvas_id)
            self._canvas_to_scene.pop(canvas_id, None)
        self._event_bus.unsubscribe_all(scene_id)
        raise NotImplementedError(
            "Scene render-layer teardown not yet implemented. "
            "Bus subscriptions have been cleaned up."
        )

    def remove_visual(self, visual_id: UUID) -> None:
        """Clean up bus subscriptions for a visual.

        Render-layer teardown is not yet implemented.
        """
        self._event_bus.unsubscribe_all(visual_id)
        self._visual_to_scene.pop(visual_id, None)
        raise NotImplementedError(
            "Visual render-layer teardown not yet implemented. "
            "Bus subscriptions have been cleaned up."
        )

    def remove_data_store(self, data_store_id: UUID) -> None:
        """Not implemented in Phase 1."""
        raise NotImplementedError("remove_data_store is not implemented in Phase 1.")

    def get_dims_widget(self, scene_id: UUID, parent: QWidget | None = None):
        """Return a dims slider widget for a scene.

        Not implemented in Phase 1.
        """
        raise NotImplementedError("get_dims_widget is not implemented in Phase 1.")

    def on_dims_changed(
        self, scene_id: UUID, callback: Callable[[DimsState], None]
    ) -> None:
        """Register a callback fired whenever the dims for *scene_id* change.

        The callback receives a ``DimsState`` snapshot.
        """
        handle = self._event_bus.subscribe(
            DimsChangedEvent,
            lambda e: callback(e.dims_state),
            entity_id=scene_id,
            owner_id=self._id,
        )
        self._external_handles.append(handle)

    def on_camera_changed(
        self, scene_id: UUID, callback: Callable[[CameraState], None]
    ) -> None:
        """Register a callback fired whenever the camera for *scene_id* changes.

        Camera event wiring is dormant in this phase; the subscription is
        registered but will never fire until camera events are wired.
        """
        handle = self._event_bus.subscribe(
            CameraChangedEvent,
            lambda e: callback(e.camera_state),
            entity_id=scene_id,
            owner_id=self._id,
        )
        self._external_handles.append(handle)

    def on_visual_changed(self, visual_id: UUID, callback: Callable) -> None:
        """Register a callback fired whenever the appearance of *visual_id* changes.

        The callback receives the live ``MultiscaleImageVisual`` model.
        """
        handle = self._event_bus.subscribe(
            AppearanceChangedEvent,
            lambda e: callback(self.get_visual_model(e.visual_id)),
            entity_id=visual_id,
            owner_id=self._id,
        )
        self._external_handles.append(handle)
