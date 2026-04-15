"""CellierController — single entry-point for building Cellier v2 viewers."""

from __future__ import annotations

import asyncio
import contextvars
from typing import TYPE_CHECKING, Any, Callable, Literal, overload
from uuid import UUID, uuid4

import numpy as np

from cellier.v2.data.image._image_memory_store import ImageMemoryStore
from cellier.v2.events import (
    AABBChangedEvent,
    AppearanceChangedEvent,
    CameraChangedEvent,
    DimsChangedEvent,
    EventBus,
    ResliceCompletedEvent,
    ResliceStartedEvent,
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
from cellier.v2.render.visuals._lines_memory import GFXLinesMemoryVisual
from cellier.v2.render.visuals._mesh_memory import GFXMeshMemoryVisual
from cellier.v2.render.visuals._points_memory import GFXPointsMemoryVisual
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
from cellier.v2.transform import AffineTransform
from cellier.v2.viewer_model import DataManager, ViewerModel
from cellier.v2.visuals._image import MultiscaleImageVisual
from cellier.v2.visuals._image_memory import ImageMemoryAppearance, ImageVisual
from cellier.v2.visuals._lines_memory import LinesMemoryAppearance, LinesVisual
from cellier.v2.visuals._mesh_memory import (
    MeshAppearance,
    MeshPhongAppearance,
    MeshVisual,
)
from cellier.v2.visuals._points_memory import PointsMarkerAppearance, PointsVisual

if TYPE_CHECKING:
    import pathlib

    from psygnal import EmissionInfo
    from PySide6.QtWidgets import QWidget

    from cellier.v2._state import CameraState, DimsState
    from cellier.v2.data._base_data_store import BaseDataStore
    from cellier.v2.data.lines._lines_memory_store import LinesMemoryStore
    from cellier.v2.data.mesh._mesh_memory_store import MeshMemoryStore
    from cellier.v2.data.points._points_memory_store import PointsMemoryStore
    from cellier.v2.render._config import RenderManagerConfig
    from cellier.v2.visuals._image import ImageAppearance


# Appearance fields that require a reslice (not just a GPU material update).
_RESLICE_FIELDS: frozenset[str] = frozenset({"lod_bias", "force_level", "frustum_cull"})

# Context variable used by update_slice_indices / update_appearance_field to
# thread a caller-supplied source_id through the synchronous psygnal bridge.
# Default None means the bridge falls back to the controller's own ID.
_source_id_override: contextvars.ContextVar[UUID | None] = contextvars.ContextVar(
    "_source_id_override", default=None
)

# Parallel context variable for update_aabb_field / _make_aabb_handler.
_aabb_source_id_override: contextvars.ContextVar[UUID | None] = contextvars.ContextVar(
    "_aabb_source_id_override", default=None
)


class CellierController:
    """The main class for constructing and controlling a cellier visualization.

    Wraps a ViewerModel (model layer) and a RenderManager (render layer).
    """

    def __init__(
        self,
        widget_parent: QWidget | None = None,
        render_config: RenderManagerConfig | None = None,
    ) -> None:
        self._widget_parent = widget_parent
        self._model = ViewerModel(data=DataManager())
        self._render_manager = RenderManager(config=render_config)
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
        # render_modes registered per scene (determines which nodes visuals build)
        self._scene_render_modes: dict[UUID, set[Literal["2d", "3d"]]] = {}
        # Handles for externally-registered callbacks
        self._external_handles: list[SubscriptionHandle] = []
        # Camera settle
        self._settle_tasks: dict[UUID, asyncio.Task] = {}
        self._event_bus.subscribe(
            CameraChangedEvent,
            self._on_camera_changed,
            owner_id=self._id,
        )
        # Wire SliceCoordinator to DimsChangedEvent first so it invalidates
        # stale 2D caches before the controller submits new slice requests.
        coordinator = self._render_manager._slice_coordinator
        self._event_bus.subscribe(
            DimsChangedEvent,
            coordinator._on_dims_changed,
            owner_id=coordinator.id,
        )
        # Controller's own dims handler: reslice the affected scene.
        self._event_bus.subscribe(
            DimsChangedEvent,
            self._on_dims_changed_bus,
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
        render_modes: set[Literal["2d", "3d"]] | None = None,
        lighting: str = "none",
    ) -> Scene:
        """Create and register a new empty scene.

        Parameters
        ----------
        dim : str
            ``"2d"`` or ``"3d"``.  Sets the initial displayed axes.
        coordinate_system : CoordinateSystem
            World coordinate system for the scene.
        name : str
            Human-readable name.
        render_modes : set of ``"2d"`` / ``"3d"`` or None
            Which rendering modes visuals added to this scene should support.
            Defaults to ``{dim}``.  Pass ``{"2d", "3d"}`` to build both 2D
            and 3D nodes up front, enabling cheap toggling via
            ``scene.dims.selection.displayed_axes``.
        lighting : str
            ``"none"`` (default) or ``"default"``.  Pass ``"default"`` to
            add ambient and directional lights — required for
            ``MeshPhongAppearance``.

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
            slice_indices_dict = dict.fromkeys(range(ndim - 2), 0)
        else:
            raise ValueError(f"dim must be '2d' or '3d', got {dim!r}")

        if render_modes is None:
            render_modes = {dim}

        dims = DimsManager(
            coordinate_system=coordinate_system,
            selection=AxisAlignedSelection(
                displayed_axes=displayed_axes,
                slice_indices=slice_indices_dict,
            ),
        )
        scene = Scene(name=name, dims=dims)
        self._model.scenes[scene.id] = scene
        self._scene_render_modes[scene.id] = render_modes
        self._render_manager.add_scene(scene.id, lighting=lighting)
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
        gpu_budget_bytes_2d: int = ...,
        threshold: float | None = ...,
        interpolation: str = ...,
        use_brick_shader: bool = ...,
        transform: AffineTransform | None = ...,
    ) -> MultiscaleImageVisual: ...

    def add_image(
        self,
        data,
        scene_id: UUID,
        appearance,
        name: str = "image",
        block_size: int = 32,
        gpu_budget_bytes: int = 1 * 1024**3,
        gpu_budget_bytes_2d: int = 64 * 1024**2,
        threshold: float | None = None,
        interpolation: str = "linear",
        use_brick_shader: bool = False,
        transform: AffineTransform | None = None,
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
            GPU memory budget for the 3D brick cache. Only used for
            multiscale path. Default 1 GiB.
        gpu_budget_bytes_2d : int
            GPU memory budget for the 2D tile cache. Only used for
            multiscale path. Default 64 MiB.
        threshold : float
            Isosurface threshold for 3D raycast rendering. Only used for
            multiscale path. Default 0.2.
        interpolation : str
            Sampler filter ``"linear"`` or ``"nearest"``. Only used for
            multiscale path. Default ``"linear"``.
        use_brick_shader: bool
            If True, use the experimental brick shader.
            Default is False.
        transform : AffineTransform or None
            Data-to-world affine transform.  Only scale and translation are
            supported; a ``ValueError`` is raised if the linear part
            contains rotation or shear.  When ``None`` the identity
            transform is used (voxels equal world units).  Only used for
            the multiscale path (ignored for ``ImageMemoryStore``).

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
                gpu_budget_bytes_2d,
                threshold,
                interpolation,
                use_brick_shader=use_brick_shader,
                transform=transform,
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

        # ── 2. Determine render modes from scene registration ───────────
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )

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
            render_modes=render_modes,
        )

        # ── 5. Register with RenderManager and controller maps ──────────
        self._render_manager.add_visual(scene_id, gfx_visual, data, displayed_axes)
        self._visual_to_scene[visual_model.id] = scene_id

        # ── 6. Wire appearance/transform bridges and EventBus subscriptions
        self._wire_appearance(visual_model)
        self._wire_aabb(visual_model)
        self._wire_transform(visual_model, scene_id)
        self._event_bus.subscribe(
            AppearanceChangedEvent,
            gfx_visual.on_appearance_changed,
            entity_id=visual_model.id,
            owner_id=visual_model.id,
        )
        self._event_bus.subscribe(
            AABBChangedEvent,
            gfx_visual.on_aabb_changed,
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

    def add_mesh(
        self,
        data: MeshMemoryStore,
        scene_id: UUID,
        appearance: MeshAppearance,
        name: str = "mesh",
        transform: AffineTransform | None = None,
    ) -> MeshVisual:
        """Add a mesh visual to a scene.

        Parameters
        ----------
        data : MeshMemoryStore
            In-memory mesh.  Normals are auto-computed if not supplied;
            indices are coerced to int32.
        scene_id : UUID
            ID of an existing scene.
        appearance : MeshFlatAppearance | MeshPhongAppearance
            Appearance.  Use MeshPhongAppearance with lighting="default"
            on the scene for shaded rendering.
        name : str
            Human-readable label.  Default "mesh".
        transform : AffineTransform | None
            Data-to-world transform for this visual. Defaults to identity when
            ``None``.

        Returns
        -------
        MeshVisual
            The live model object.
        """
        import warnings

        # Warn if Phong requested but no lights in scene.
        if isinstance(appearance, MeshPhongAppearance):
            import pygfx as gfx

            sm = self._render_manager._scenes[scene_id]
            has_light = any(
                isinstance(c, (gfx.AmbientLight, gfx.DirectionalLight))
                for c in sm.scene.children
            )
            if not has_light:
                warnings.warn(
                    "MeshPhongAppearance requires lights in the scene. "
                    "Pass lighting='default' to add_scene(), otherwise "
                    "the mesh will render black.",
                    stacklevel=2,
                )

        if data.id not in self._model.data.stores:
            self._model.data.stores[data.id] = data

        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        resolved_transform = (
            transform
            if transform is not None
            else AffineTransform.identity(ndim=data.positions.shape[1])
        )

        visual_model = MeshVisual(
            name=name,
            data_store_id=str(data.id),
            appearance=appearance,
            transform=resolved_transform,
        )
        scene.visuals.append(visual_model)

        gfx_visual = GFXMeshMemoryVisual(
            visual_model=visual_model,
            render_modes=render_modes,
            transform=resolved_transform,
        )

        self._render_manager.add_visual(scene_id, gfx_visual, data, displayed_axes)
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

    def add_points(
        self,
        data: PointsMemoryStore,
        scene_id: UUID,
        appearance: PointsMarkerAppearance | None = None,
        name: str = "points",
        transform: AffineTransform | None = None,
    ) -> PointsVisual:
        """Add a points visual backed by a PointsMemoryStore.

        Parameters
        ----------
        data : PointsMemoryStore
            The backing data store.  Registered in the model if not already
            present.
        scene_id : UUID
            ID of the target scene.
        appearance : PointsMarkerAppearance | None
            Appearance model.  Defaults to PointsMarkerAppearance() if None.
        name : str
            Human-readable label for the visual.
        transform : AffineTransform | None
            Data-to-world transform for this visual. Defaults to identity when
            ``None``.

        Returns
        -------
        PointsVisual
            The newly created model-layer visual.
        """
        if appearance is None:
            appearance = PointsMarkerAppearance()

        if data.id not in self._model.data.stores:
            self._model.data.stores[data.id] = data

        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        resolved_transform = (
            transform
            if transform is not None
            else AffineTransform.identity(ndim=data.ndim)
        )

        visual_model = PointsVisual(
            name=name,
            data_store_id=str(data.id),
            appearance=appearance,
            transform=resolved_transform,
        )
        scene.visuals.append(visual_model)

        gfx_visual = GFXPointsMemoryVisual(
            visual_model=visual_model,
            render_modes=render_modes,
            transform=resolved_transform,
        )

        self._render_manager.add_visual(scene_id, gfx_visual, data, displayed_axes)
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

    def add_lines(
        self,
        data: LinesMemoryStore,
        scene_id: UUID,
        appearance: LinesMemoryAppearance | None = None,
        name: str = "lines",
        transform: AffineTransform | None = None,
    ) -> LinesVisual:
        """Add a lines visual backed by a LinesMemoryStore.

        Parameters
        ----------
        data : LinesMemoryStore
            The backing data store.  Registered in the model if not already
            present.
        scene_id : UUID
            ID of the target scene.
        appearance : LinesMemoryAppearance | None
            Appearance model.  Defaults to LinesMemoryAppearance() if None.
        name : str
            Human-readable label for the visual.
        transform : AffineTransform | None
            Data-to-world transform for this visual. Defaults to identity when
            ``None``.

        Returns
        -------
        LinesVisual
            The newly created model-layer visual.
        """
        if appearance is None:
            appearance = LinesMemoryAppearance()

        if data.id not in self._model.data.stores:
            self._model.data.stores[data.id] = data

        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        resolved_transform = (
            transform
            if transform is not None
            else AffineTransform.identity(ndim=data.ndim)
        )

        visual_model = LinesVisual(
            name=name,
            data_store_id=str(data.id),
            appearance=appearance,
            transform=resolved_transform,
        )
        scene.visuals.append(visual_model)

        gfx_visual = GFXLinesMemoryVisual(
            visual_model=visual_model,
            render_modes=render_modes,
            transform=resolved_transform,
        )

        self._render_manager.add_visual(scene_id, gfx_visual, data, displayed_axes)
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

    def _add_image_multiscale(
        self,
        data: BaseDataStore,
        scene_id: UUID,
        appearance: ImageAppearance,
        name: str,
        block_size: int,
        gpu_budget_bytes: int,
        gpu_budget_bytes_2d: int,
        threshold: float | None,
        interpolation: str,
        use_brick_shader: bool = False,
        transform: AffineTransform | None = None,
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

        if transform is not None:
            visual_model.transform = transform

        scene.visuals.append(visual_model)

        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )

        level_shapes = list(data.level_shapes)
        gfx_visual = GFXMultiscaleImageVisual.from_cellier_model(
            model=visual_model,
            level_shapes=level_shapes,
            render_modes=render_modes,
            displayed_axes=displayed_axes,
            block_size=block_size,
            gpu_budget_bytes=gpu_budget_bytes,
            gpu_budget_bytes_2d=gpu_budget_bytes_2d,
            interpolation=interpolation,
            use_brick_shader=use_brick_shader,
        )

        self._render_manager.add_visual(scene_id, gfx_visual, data, displayed_axes)
        self._visual_to_scene[visual_model.id] = scene_id
        self._wire_appearance(visual_model)
        self._wire_aabb(visual_model)
        self._wire_transform(visual_model, scene_id)
        self._event_bus.subscribe(
            AppearanceChangedEvent,
            gfx_visual.on_appearance_changed,
            entity_id=visual_model.id,
            owner_id=visual_model.id,
        )
        self._event_bus.subscribe(
            AABBChangedEvent,
            gfx_visual.on_aabb_changed,
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
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )
        level_shapes = list(data_store.level_shapes)
        gfx_visual = GFXMultiscaleImageVisual.from_cellier_model(
            model=visual_model,
            level_shapes=level_shapes,
            render_modes=render_modes,
            displayed_axes=displayed_axes,
            block_size=block_size,
            gpu_budget_bytes=gpu_budget_bytes,
            threshold=threshold,
            interpolation=interpolation,
        )
        self._render_manager.add_visual(
            scene_id, gfx_visual, data_store, displayed_axes
        )
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

        return canvas_view.widget

    # ------------------------------------------------------------------
    # Scene and visual lookup
    # ------------------------------------------------------------------

    def get_scene(self, scene_id: UUID) -> Scene:
        """Return the live Scene model for scene_id."""
        return self._model.scenes[scene_id]

    def fit_camera(self, scene_id: UUID) -> None:
        """Fit the camera to the current scene bounding box.

        Safe to call immediately after ``add_image`` and transform assignment
        — the node matrix is set at construction time so no chunk data needs
        to be loaded first.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene whose camera should be fitted.
        """
        canvas = self._render_manager._find_canvas_for_scene(scene_id)
        if canvas is None:
            return
        gfx_scene = self._render_manager.get_scene(scene_id)
        canvas.show_object(gfx_scene)

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
            else:
                configs[visual.id] = VisualRenderConfig()
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
                self._switch_canvas_cameras(
                    scene_id, new_state.selection.displayed_axes
                )
            source_id = _source_id_override.get() or self._id
            self._event_bus.emit(
                DimsChangedEvent(
                    source_id=source_id,
                    scene_id=scene_id,
                    dims_state=new_state,
                    displayed_axes_changed=displayed_axes_changed,
                )
            )

        return _on_dims_psygnal

    def _rebuild_visuals_geometry(
        self, scene_id: UUID, displayed_axes: tuple[int, ...]
    ) -> None:
        """Swap each visual's active node after a displayed_axes change.

        Calls ``gfx_visual.get_node_for_dims(displayed_axes)`` on each visual
        to obtain the new node, then delegates scene-graph surgery to
        ``SceneManager.swap_node``.  Single-node visuals (mesh, lines) are
        handled transparently because ``swap_node`` no-ops when the old and
        new nodes are the same object.
        """
        scene = self._model.scenes[scene_id]
        scene_manager = self._render_manager._scenes[scene_id]

        for visual_model in scene.visuals:
            gfx_visual = scene_manager.get_visual(visual_model.id)
            new_node = gfx_visual.get_node_for_dims(displayed_axes)
            scene_manager.swap_node(visual_model.id, new_node)

    def _switch_canvas_cameras(
        self, scene_id: UUID, displayed_axes: tuple[int, ...]
    ) -> None:
        """Switch cameras on all canvases attached to *scene_id*.

        Calls ``show_object`` on canvases that are being activated for the
        first time in the new dimensionality.
        """
        new_dim = "3d" if len(displayed_axes) == 3 else "2d"
        gfx_scene = self._render_manager.get_scene(scene_id)
        for canvas_id in self._scene_to_canvases.get(scene_id, []):
            canvas_view = self._render_manager._canvases[canvas_id]
            first_visit = canvas_view.switch_dim(new_dim)
            if first_visit:
                canvas_view.show_object(gfx_scene)

    def _wire_transform(
        self,
        visual: MultiscaleImageVisual
        | ImageVisual
        | MeshVisual
        | PointsVisual
        | LinesVisual,
        scene_id: UUID,
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

    def _wire_appearance(
        self,
        visual: MultiscaleImageVisual
        | ImageVisual
        | MeshVisual
        | PointsVisual
        | LinesVisual,
    ) -> None:
        """Subscribe to all field changes on a visual's appearance model."""
        visual.appearance.events.connect(self._make_appearance_handler(visual.id))

    def _wire_aabb(self, visual: MultiscaleImageVisual | ImageVisual) -> None:
        """Subscribe to all field changes on a visual's aabb model."""
        visual.aabb.events.connect(self._make_aabb_handler(visual.id))

    def _make_aabb_handler(self, visual_id: UUID) -> Callable:
        """Return a psygnal catch-all handler for a visual's AABBParams."""

        def _on_aabb_psygnal(info: EmissionInfo) -> None:
            field_name: str = info.signal.name
            new_value = info.args[0]
            source_id = _aabb_source_id_override.get() or self._id
            self._event_bus.emit(
                AABBChangedEvent(
                    source_id=source_id,
                    visual_id=visual_id,
                    field_name=field_name,
                    new_value=new_value,
                )
            )

        return _on_aabb_psygnal

    def _make_appearance_handler(self, visual_id: UUID) -> Callable:
        """Return a psygnal catch-all handler for a visual's ImageAppearance."""

        def _on_appearance_psygnal(info: EmissionInfo) -> None:
            field_name: str = info.signal.name
            new_value = info.args[0]
            source_id = _source_id_override.get() or self._id
            if field_name == "visible":
                self._event_bus.emit(
                    VisualVisibilityChangedEvent(
                        source_id=source_id,
                        visual_id=visual_id,
                        visible=new_value,
                    )
                )
            else:
                self._event_bus.emit(
                    AppearanceChangedEvent(
                        source_id=source_id,
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

    def _on_dims_changed_bus(self, event: DimsChangedEvent) -> None:
        """Bus handler — reslice the scene whenever its dims state changes."""
        self.reslice_scene(event.scene_id)

    # ------------------------------------------------------------------
    # Model mutation with source-ID threading
    # ------------------------------------------------------------------

    def update_slice_indices(
        self,
        scene_id: UUID,
        slice_indices: dict[int, int],
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set ``slice_indices`` on a scene's dims.

        Tags the emitted bus event with *source_id*.
        GUI widgets should pass ``source_id=self._id`` so their own
        ``DimsChangedEvent`` subscription can ignore the echo.

        Parameters
        ----------
        scene_id :
            Target scene.
        slice_indices :
            Mapping of axis index → slice position.
        source_id :
            UUID to stamp on the emitted ``DimsChangedEvent``.  Defaults
            to the controller's own ID.
        """
        token = _source_id_override.set(source_id)
        try:
            self._model.scenes[scene_id].dims.selection.slice_indices = slice_indices
        finally:
            _source_id_override.reset(token)

    def update_appearance_field(
        self,
        visual_id: UUID,
        field: str,
        value: Any,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set one field on a visual's appearance model.

        Tags the emitted bus event with *source_id*.
        GUI widgets should pass ``source_id=self._id`` so their own
        ``AppearanceChangedEvent`` subscription can ignore the echo.

        Parameters
        ----------
        visual_id :
            Target visual.
        field :
            Attribute name on the appearance model, e.g. ``"clim"``.
        value :
            New value for the field.
        source_id :
            UUID to stamp on the emitted ``AppearanceChangedEvent``.  Defaults
            to the controller's own ID.
        """
        visual = self.get_visual_model(visual_id)
        token = _source_id_override.set(source_id)
        try:
            setattr(visual.appearance, field, value)
        finally:
            _source_id_override.reset(token)

    def update_aabb_field(
        self,
        visual_id: UUID,
        field: str,
        value: Any,
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set one field on a visual's AABB params model.

        Tags the emitted bus event with *source_id*.
        GUI widgets should pass ``source_id=self._id`` so their own
        ``AABBChangedEvent`` subscription can ignore the echo.

        Parameters
        ----------
        visual_id :
            Target visual.
        field :
            Attribute name on the AABB model, e.g. ``"enabled"``.
        value :
            New value for the field.
        source_id :
            UUID to stamp on the emitted ``AABBChangedEvent``.  Defaults
            to the controller's own ID.
        """
        visual = self.get_visual_model(visual_id)
        token = _aabb_source_id_override.set(source_id)
        try:
            setattr(visual.aabb, field, value)
        finally:
            _aabb_source_id_override.reset(token)

    # ------------------------------------------------------------------
    # Camera settle
    # ------------------------------------------------------------------

    @property
    def camera_reslice_enabled(self) -> bool:
        """Whether camera movement triggers automatic reslicing."""
        return self._render_manager.config.camera.reslice_enabled

    @camera_reslice_enabled.setter
    def camera_reslice_enabled(self, value: bool) -> None:
        self._render_manager.config.camera.reslice_enabled = value
        if not value:
            for task in self._settle_tasks.values():
                if not task.done():
                    task.cancel()
            self._settle_tasks.clear()

    @property
    def camera_settle_threshold_s(self) -> float:
        """Debounce delay before reslice after camera movement."""
        return self._render_manager.config.camera.settle_threshold_s

    @camera_settle_threshold_s.setter
    def camera_settle_threshold_s(self, value: float) -> None:
        self._render_manager.config.camera.settle_threshold_s = value

    def _on_camera_changed(self, event: CameraChangedEvent) -> None:
        """Synchronous bus handler — updates camera model and schedules settle task."""
        self._update_camera_model(event.scene_id, event.camera_state)

        if not self._render_manager.config.camera.reslice_enabled:
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
            self._render_manager.config.camera.settle_threshold_s,
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
            await asyncio.sleep(self._render_manager.config.camera.settle_threshold_s)
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

    def add_labels(self, *args, **kwargs):
        """Not implemented in Phase 1."""
        raise NotImplementedError("add_labels is not implemented in Phase 1.")

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

    # ------------------------------------------------------------------
    # External event subscriptions
    # ------------------------------------------------------------------

    def on_dims_changed(
        self,
        scene_id: UUID,
        callback: Callable[[DimsChangedEvent], None],
        *,
        owner_id: UUID | None = None,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired whenever the dims for *scene_id* change.

        The callback receives the full ``DimsChangedEvent``, which includes
        ``source_id`` for echo-filtering and ``dims_state`` for the new state.

        Parameters
        ----------
        scene_id :
            The scene to watch.
        callback :
            Called with the ``DimsChangedEvent`` on each dims change.
        owner_id :
            When provided, the subscription is registered under this UUID so
            that ``unsubscribe_owner(owner_id)`` removes it.  Pass a widget's
            own UUID for per-widget cleanup.  Defaults to the controller's ID.
        weak :
            If True, hold only a weak reference to *callback*.  Use for
            transient widgets that may be destroyed outside the controller's
            teardown path.  Cannot be used with lambdas.

        Returns
        -------
        SubscriptionHandle
            Pass to ``EventBus.unsubscribe()`` for individual removal.
        """
        effective_owner = owner_id if owner_id is not None else self._id
        handle = self._event_bus.subscribe(
            DimsChangedEvent,
            callback,
            entity_id=scene_id,
            owner_id=effective_owner,
            weak=weak,
        )
        if owner_id is None:
            self._external_handles.append(handle)
        return handle

    def on_camera_changed(
        self,
        scene_id: UUID,
        callback: Callable[[CameraChangedEvent], None],
        *,
        owner_id: UUID | None = None,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired whenever the camera for *scene_id* changes.

        Camera event wiring is dormant in this phase; the subscription is
        registered but will never fire until camera events are wired.

        Parameters
        ----------
        scene_id :
            The scene to watch.
        callback :
            Called with the ``CameraChangedEvent`` on each camera change.
        owner_id :
            Per-widget owner UUID for bulk cleanup via ``unsubscribe_owner``.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        effective_owner = owner_id if owner_id is not None else self._id
        handle = self._event_bus.subscribe(
            CameraChangedEvent,
            callback,
            entity_id=scene_id,
            owner_id=effective_owner,
            weak=weak,
        )
        if owner_id is None:
            self._external_handles.append(handle)
        return handle

    def on_visual_changed(
        self,
        visual_id: UUID,
        callback: Callable[[AppearanceChangedEvent], None],
        *,
        owner_id: UUID | None = None,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired whenever the appearance of *visual_id* changes.

        The callback receives the full ``AppearanceChangedEvent``, which
        includes ``source_id`` for echo-filtering, ``field_name``, and
        ``new_value``.

        Parameters
        ----------
        visual_id :
            The visual to watch.
        callback :
            Called with the ``AppearanceChangedEvent`` on each appearance change.
        owner_id :
            Per-widget owner UUID for bulk cleanup via ``unsubscribe_owner``.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        effective_owner = owner_id if owner_id is not None else self._id
        handle = self._event_bus.subscribe(
            AppearanceChangedEvent,
            callback,
            entity_id=visual_id,
            owner_id=effective_owner,
            weak=weak,
        )
        if owner_id is None:
            self._external_handles.append(handle)
        return handle

    def on_aabb_changed(
        self,
        visual_id: UUID,
        callback: Callable[[AABBChangedEvent], None],
        *,
        owner_id: UUID | None = None,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired whenever the AABB params of *visual_id* change.

        The callback receives the full ``AABBChangedEvent``, which includes
        ``source_id`` for echo-filtering, ``field_name``, and ``new_value``.

        Parameters
        ----------
        visual_id :
            The visual to watch.
        callback :
            Called with the ``AABBChangedEvent`` on each AABB change.
        owner_id :
            Per-widget owner UUID for bulk cleanup via ``unsubscribe_owner``.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        effective_owner = owner_id if owner_id is not None else self._id
        handle = self._event_bus.subscribe(
            AABBChangedEvent,
            callback,
            entity_id=visual_id,
            owner_id=effective_owner,
            weak=weak,
        )
        if owner_id is None:
            self._external_handles.append(handle)
        return handle

    def on_reslice_started(
        self,
        scene_id: UUID,
        callback: Callable[[ResliceStartedEvent], None],
        *,
        owner_id: UUID | None = None,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired when a reslice cycle begins for *scene_id*.

        Useful for showing a loading indicator.  Fired once per reslice
        submission, before any async data fetching starts.

        Parameters
        ----------
        scene_id :
            The scene to watch.
        callback :
            Called with the ``ResliceStartedEvent``.
        owner_id :
            Per-widget owner UUID for bulk cleanup via ``unsubscribe_owner``.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        effective_owner = owner_id if owner_id is not None else self._id
        handle = self._event_bus.subscribe(
            ResliceStartedEvent,
            callback,
            entity_id=scene_id,
            owner_id=effective_owner,
            weak=weak,
        )
        if owner_id is None:
            self._external_handles.append(handle)
        return handle

    def on_reslice_completed(
        self,
        visual_id: UUID,
        callback: Callable[[ResliceCompletedEvent], None],
        *,
        owner_id: UUID | None = None,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired when a reslice cycle completes for *visual_id*.

        Useful for hiding a loading indicator.  Fired once per visual per
        reslice cycle, after all bricks/tiles in the batch are committed.

        Parameters
        ----------
        visual_id :
            The visual to watch.
        callback :
            Called with the ``ResliceCompletedEvent``.
        owner_id :
            Per-widget owner UUID for bulk cleanup via ``unsubscribe_owner``.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        effective_owner = owner_id if owner_id is not None else self._id
        handle = self._event_bus.subscribe(
            ResliceCompletedEvent,
            callback,
            entity_id=visual_id,
            owner_id=effective_owner,
            weak=weak,
        )
        if owner_id is None:
            self._external_handles.append(handle)
        return handle

    def unsubscribe_owner(self, owner_id: UUID) -> None:
        """Remove all subscriptions registered under *owner_id*.

        GUI widgets should call this from their Qt ``closeEvent`` or
        ``destroyed`` signal handler to deterministically clean up their
        bus subscriptions.

        Parameters
        ----------
        owner_id :
            The UUID used as ``owner_id`` when the subscriptions were
            registered (typically the widget's own ``self._id``).
        """
        self._event_bus.unsubscribe_all(owner_id)
