"""CellierController is the coordinator between the models and the rendered views."""

from __future__ import annotations

import asyncio
import contextvars
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Generator, Literal
from uuid import UUID, uuid4

import numpy as np

from cellier.v2.events import (
    AABBChangedEvent,
    AABBUpdateEvent,
    AppearanceChangedEvent,
    AppearanceUpdateEvent,
    CameraChangedEvent,
    DimsChangedEvent,
    DimsUpdateEvent,
    EventBus,
    ResliceCompletedEvent,
    ResliceStartedEvent,
    SceneAddedEvent,
    SceneRemovedEvent,
    SubscriptionHandle,
    SubscriptionSpec,
    TransformChangedEvent,
    VisualAddedEvent,
    VisualRemovedEvent,
    VisualVisibilityChangedEvent,
)
from cellier.v2.events._events import (
    CanvasMouseMove2DEvent,
    CanvasMouseMove3DEvent,
    CanvasMousePress2DEvent,
    CanvasMousePress3DEvent,
    CanvasMouseRelease2DEvent,
    CanvasMouseRelease3DEvent,
    _CanvasRawPointerEvent,
)
from cellier.v2.logging import _CAMERA_LOGGER, _SOURCE_ID_LOGGER
from cellier.v2.render._scene_config import VisualRenderConfig
from cellier.v2.render.render_manager import RenderManager
from cellier.v2.render.visuals._canvas_overlay import GFXCenteredAxes2D
from cellier.v2.render.visuals._image import GFXMultiscaleImageVisual
from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual
from cellier.v2.render.visuals._image_memory_multichannel import (
    GFXMultichannelImageMemoryVisual,
)
from cellier.v2.render.visuals._image_multiscale_multichannel import (
    GFXMultichannelMultiscaleImageVisual,
)
from cellier.v2.render.visuals._lines_memory import GFXLinesMemoryVisual
from cellier.v2.render.visuals._mesh_memory import GFXMeshMemoryVisual
from cellier.v2.render.visuals._points_memory import GFXPointsMemoryVisual
from cellier.v2.scene.cameras import (
    CameraType,
    OrbitCameraController,
    OrthographicCamera,
    PanZoomCameraController,
    PerspectiveCamera,
)
from cellier.v2.scene.canvas import Canvas
from cellier.v2.scene.dims import AxisAlignedSelection, CoordinateSystem, DimsManager
from cellier.v2.scene.scene import Scene
from cellier.v2.transform import AffineTransform
from cellier.v2.viewer_model import DataManager, ViewerModel
from cellier.v2.visuals._canvas_overlay import CenteredAxes2D
from cellier.v2.visuals._image import (
    ImageAppearance,
    MultichannelMultiscaleImageVisual,
    MultiscaleImageRenderConfig,
    MultiscaleImageVisual,
)
from cellier.v2.visuals._image_memory import (
    ImageMemoryAppearance,
    ImageVisual,
    MultichannelImageVisual,
)
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
    from cellier.v2.data.image._image_memory_store import ImageMemoryStore
    from cellier.v2.data.lines._lines_memory_store import LinesMemoryStore
    from cellier.v2.data.mesh._mesh_memory_store import MeshMemoryStore
    from cellier.v2.data.points._points_memory_store import PointsMemoryStore
    from cellier.v2.render._config import RenderManagerConfig
    from cellier.v2.render.canvas_view import CanvasView
    from cellier.v2.visuals._canvas_overlay import CanvasOverlay
    from cellier.v2.visuals._channel_appearance import ChannelAppearance
    from cellier.v2.visuals._types import VisualType


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

    Wraps a ViewerModel (model layer) and a RenderManager (render layer) and
    performs the synchronization between both.
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
        # Event buses
        self._id: UUID = uuid4()
        self._event_bus: EventBus = EventBus()
        self._incoming_events: EventBus = EventBus()
        # Cache of last-known displayed_axes per scene for change detection
        self._dims_cache: dict[UUID, tuple[int, ...]] = {}
        # render_modes registered per scene (determines which nodes visuals build)
        self._scene_render_modes: dict[UUID, set[Literal["2d", "3d"]]] = {}
        # Camera settle
        self._settle_tasks: dict[UUID, asyncio.Task] = {}
        # Stored psygnal bridge handlers keyed by visual_id.
        # Each entry is a list of (signal, handler) pairs.  psygnal
        # disconnect() requires the exact handler object; closures are not
        # equality-comparable by value, so references must be retained.
        # Storing the signal alongside the handler avoids branching on visual
        # type during teardown.
        self._visual_psygnal_handlers: dict[UUID, list[tuple]] = {}
        # When True, transform-change handlers skip reslice_scene.  Managed
        # by the suppress_reslice context manager.  This is a flat boolean, so
        # nested suppress_reslice calls or concurrent async transform mutations
        # will interfere — use a depth counter if that ever becomes necessary.
        self._suppress_reslice: bool = False
        self._event_bus.subscribe(
            CameraChangedEvent,
            self._on_camera_changed,
            owner_id=self._id,
        )
        # Must be called before subscribing _on_dims_changed_bus below so the
        # SliceCoordinator invalidates stale 2D caches before the controller
        # submits new slice requests.
        self._render_manager.connect_event_bus(self._event_bus)
        # Controller's own dims handler: reslice the affected scene.
        self._event_bus.subscribe(
            DimsChangedEvent,
            self._on_dims_changed_bus,
            owner_id=self._id,
        )
        # Subscribe to internal raw pointer events emitted by RenderManager.
        self._event_bus.subscribe(
            _CanvasRawPointerEvent,
            self._on_raw_pointer_event,
            owner_id=self._id,
        )
        # Incoming bus: GUI update events dispatched to update_* methods.
        self._incoming_events.subscribe(
            AppearanceUpdateEvent,
            self._on_appearance_update,
            owner_id=self._id,
        )
        self._incoming_events.subscribe(
            DimsUpdateEvent,
            self._on_dims_update,
            owner_id=self._id,
        )
        self._incoming_events.subscribe(
            AABBUpdateEvent,
            self._on_aabb_update,
            owner_id=self._id,
        )

    @property
    def incoming_events(self) -> EventBus:
        """Incoming event bus for GUI-driven model mutations.

        Emit ``AppearanceUpdateEvent``, ``DimsUpdateEvent``, or
        ``AABBUpdateEvent`` onto this bus to request model changes.
        The controller dispatches each event to the corresponding
        ``update_*`` method, preserving ``source_id`` end-to-end.
        """
        return self._incoming_events

    def set_widget_parent(self, parent: QWidget) -> None:
        """Set the Qt parent for subsequently created canvas widgets.

        TODO: in the future, we should have an abstraction for different GUI backends.
        """
        self._widget_parent = parent

    # ------------------------------------------------------------------
    # Construction class methods (stubs)
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: ViewerModel,
        widget_parent: QWidget | None = None,
        render_config: RenderManagerConfig | None = None,
    ) -> CellierController:
        """Construct a controller from a serialized ViewerModel.

        Iteratively adds all data stores, scenes, visuals, and canvases
        through the public API. The order is:

        1. Data stores  — registered before visuals reference them.
        2. Scenes       — registered with render_modes and lighting from model.
        3. Visuals      — added per scene; data stores must already be present.
        4. Canvases     — restored with camera state from model.

        Parameters
        ----------
        model : ViewerModel
            A ViewerModel loaded from disk or constructed programmatically.
        widget_parent : QWidget or None
            Qt parent for canvas widgets. Defaults to None.
        render_config : RenderManagerConfig or None
            Render pipeline configuration. Defaults to None (uses defaults).

        Returns
        -------
        CellierController
        """
        controller = cls(
            widget_parent=widget_parent,
            render_config=render_config,
        )

        # 1. Register all data stores.
        for store in model.data.stores.values():
            controller.add_data_store(store)

        # 2. Register all scenes (render_modes and lighting come from the model).
        for scene in model.scenes.values():
            controller.add_scene_model(scene)

        # 3. Add visuals per scene. Data stores are already registered in step 1.
        # The deserialized model's scene.visuals already contains the models.
        # Clear them first so _add_* helpers can re-append with proper wiring.
        for scene in model.scenes.values():
            visual_models = list(scene.visuals)
            scene.visuals.clear()
            for visual_model in visual_models:
                controller.add_visual(scene.id, visual_model)

        # 4. Restore canvases with camera state from the model.
        for scene in model.scenes.values():
            for canvas_model in scene.canvases.values():
                controller.add_canvas_model(scene.id, canvas_model)

        return controller

    @classmethod
    def from_file(
        cls,
        path: str | pathlib.Path,
        widget_parent: QWidget | None = None,
        render_config: RenderManagerConfig | None = None,
    ) -> CellierController:
        """Deserialize a ViewerModel from disk and construct a controller.

        Parameters
        ----------
        path : str or Path
            Path to a JSON file previously written by ``to_file``.
        widget_parent : QWidget or None
            Qt parent for canvas widgets.
        render_config : RenderManagerConfig or None
            Render pipeline configuration.

        Returns
        -------
        CellierController
        """
        model = ViewerModel.from_file(path)
        return cls.from_model(
            model,
            widget_parent=widget_parent,
            render_config=render_config,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_file(self, path: str | pathlib.Path) -> None:
        """Serialize the current model state to a JSON file."""
        self._model.to_file(path)

    def to_model(self) -> ViewerModel:
        """Return a copy of the current model state."""
        return self._model.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Scene management
    # ------------------------------------------------------------------

    def add_scene_model(self, scene: Scene) -> Scene:
        """Register a pre-built Scene model with the controller.

        Used by ``from_model`` to restore scenes from a serialized
        ViewerModel, and called internally by ``add_scene``.

        Parameters
        ----------
        scene : Scene
            Pre-built scene model.

        Returns
        -------
        Scene
            The same object passed in.
        """
        self._model.scenes[scene.id] = scene
        self._scene_render_modes[scene.id] = scene.render_modes
        self._render_manager.add_scene(scene.id, lighting=scene.lighting)
        self._scene_to_canvases[scene.id] = []
        self._wire_dims_model(scene)
        self._event_bus.emit(SceneAddedEvent(source_id=self._id, scene_id=scene.id))
        return scene

    def add_scene(
        self,
        *,
        name: str = "scene",
        dim: Literal["2d", "3d"] = "3d",
        coordinate_system: CoordinateSystem | None = None,
        render_modes: set[Literal["2d", "3d"]] | None = None,
        lighting: Literal["none", "default"] = "none",
    ) -> Scene:
        """Create a Scene from keyword arguments and register it.

        Parameters
        ----------
        name : str
            Human-readable scene name.
        dim : "2d" or "3d"
            Initial display dimensionality.  ``"3d"`` sets
            ``displayed_axes`` to the last three axes of the coordinate
            system; ``"2d"`` sets it to the last two.
        coordinate_system : CoordinateSystem or None
            World coordinate system.  Defaults to a 3-axis ``("z", "y", "x")``
            system when ``None``.
        render_modes : set or None
            Which rendering modes visuals should support.  Defaults to
            ``{"2d", "3d"}``.
        lighting : "none" or "default"
            Pass ``"default"`` to add ambient/directional lights (required for
            ``MeshPhongAppearance``).

        Returns
        -------
        Scene
            The newly created and registered Scene.
        """
        if coordinate_system is None:
            coordinate_system = CoordinateSystem(
                name="world", axis_labels=("z", "y", "x")
            )
        ndim = len(coordinate_system.axis_labels)
        n_displayed = 3 if dim == "3d" else 2
        if ndim < n_displayed:
            raise ValueError(
                f"coordinate_system has {ndim} axes but dim={dim!r} requires "
                f"at least {n_displayed}."
            )
        displayed_axes = tuple(range(ndim - n_displayed, ndim))
        slice_indices = {i: 0 for i in range(ndim) if i not in displayed_axes}
        dims = DimsManager(
            coordinate_system=coordinate_system,
            selection=AxisAlignedSelection(
                displayed_axes=displayed_axes,
                slice_indices=slice_indices,
            ),
        )
        scene = Scene(
            name=name,
            dims=dims,
            render_modes=render_modes if render_modes is not None else {"2d", "3d"},
            lighting=lighting,
        )
        return self.add_scene_model(scene)

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
    # Visual management — public API
    # ------------------------------------------------------------------

    def add_visual(
        self,
        scene_id: UUID,
        visual_model: VisualType,
        data_store: BaseDataStore | None = None,
    ) -> VisualType:
        """Register a pre-built visual model with a scene.

        This is the canonical construction path used by ``from_model``.
        All typed convenience methods (``add_image``, ``add_mesh``, etc.)
        delegate to this method internally.

        Parameters
        ----------
        scene_id : UUID
            ID of an existing scene.
        visual_model : VisualType
            Pre-built visual model. Its ``data_store_id`` must already be
            registered via ``add_data_store``, or ``data_store`` must be
            passed explicitly.
        data_store : BaseDataStore or None
            If provided, register the store first (no-op if already present),
            then use it. If ``None``, the store is looked up by
            ``visual_model.data_store_id``; a ``KeyError`` is raised if not
            found.

        Returns
        -------
        VisualType
            The same visual_model passed in.

        Raises
        ------
        KeyError
            If ``data_store`` is None and ``visual_model.data_store_id`` is not
            registered.
        TypeError
            If the visual type is not recognized.
        """
        if data_store is not None:
            if data_store.id not in self._model.data.stores:
                self._model.data.stores[data_store.id] = data_store

        if isinstance(visual_model, MultiscaleImageVisual):
            return self._add_multiscale_image_visual(scene_id, visual_model)
        elif isinstance(visual_model, ImageVisual):
            return self._add_image_visual(scene_id, visual_model)
        elif isinstance(visual_model, MultichannelImageVisual):
            return self._add_multichannel_image_memory_visual(scene_id, visual_model)
        elif isinstance(visual_model, MultichannelMultiscaleImageVisual):
            return self._add_multichannel_multiscale_image_visual(
                scene_id, visual_model
            )
        elif isinstance(visual_model, PointsVisual):
            return self._add_points_visual(scene_id, visual_model)
        elif isinstance(visual_model, LinesVisual):
            return self._add_lines_visual(scene_id, visual_model)
        elif isinstance(visual_model, MeshVisual):
            return self._add_mesh_visual(scene_id, visual_model)
        else:
            raise TypeError(
                f"Unrecognized visual type {type(visual_model)!r}. "
                "Register a handler in add_visual."
            )

    def add_image(
        self,
        data: ImageMemoryStore,
        scene_id: UUID,
        appearance: ImageMemoryAppearance,
        name: str = "image",
    ) -> ImageVisual:
        """Add an in-memory image visual to a scene.

        Parameters
        ----------
        data : ImageMemoryStore
            The backing data store.
        scene_id : UUID
            ID of an existing scene.
        appearance : ImageMemoryAppearance
            Appearance parameters.
        name : str
            Human-readable label. Default ``"image"``.

        Returns
        -------
        ImageVisual
        """
        visual_model = ImageVisual(
            name=name,
            data_store_id=str(data.id),
            appearance=appearance,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

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
            Appearance.  Use MeshPhongAppearance with ``lighting="default"``
            on the scene for shaded rendering.
        name : str
            Human-readable label.  Default ``"mesh"``.
        transform : AffineTransform or None
            Data-to-world transform for this visual. Defaults to identity when
            ``None``.

        Returns
        -------
        MeshVisual
        """
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
        return self.add_visual(scene_id, visual_model, data_store=data)

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
            The backing data store.
        scene_id : UUID
            ID of the target scene.
        appearance : PointsMarkerAppearance or None
            Appearance model.  Defaults to PointsMarkerAppearance() if None.
        name : str
            Human-readable label for the visual.
        transform : AffineTransform or None
            Data-to-world transform for this visual. Defaults to identity when
            ``None``.

        Returns
        -------
        PointsVisual
        """
        if appearance is None:
            appearance = PointsMarkerAppearance()

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
        return self.add_visual(scene_id, visual_model, data_store=data)

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
            The backing data store.
        scene_id : UUID
            ID of the target scene.
        appearance : LinesMemoryAppearance or None
            Appearance model.  Defaults to LinesMemoryAppearance() if None.
        name : str
            Human-readable label for the visual.
        transform : AffineTransform or None
            Data-to-world transform for this visual. Defaults to identity when
            ``None``.

        Returns
        -------
        LinesVisual
        """
        if appearance is None:
            appearance = LinesMemoryAppearance()

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
        return self.add_visual(scene_id, visual_model, data_store=data)

    def add_image_multiscale(
        self,
        data: BaseDataStore,
        scene_id: UUID,
        appearance: ImageAppearance,
        name: str = "image",
        render_config: MultiscaleImageRenderConfig | None = None,
        transform: AffineTransform | None = None,
    ) -> MultiscaleImageVisual:
        """Add a multiscale image visual to a scene.

        Parameters
        ----------
        data : BaseDataStore
            The backing data store.
        scene_id : UUID
            ID of an existing scene.
        appearance : ImageAppearance
            Visual appearance parameters.
        name : str
            Human-readable label. Default ``"image"``.
        render_config : MultiscaleImageRenderConfig or None
            Render-layer configuration. Defaults to
            ``MultiscaleImageRenderConfig()`` with all default values if None.
        transform : AffineTransform or None
            Data-to-world transform. Defaults to identity when None.

        Returns
        -------
        MultiscaleImageVisual
        """
        if render_config is None:
            render_config = MultiscaleImageRenderConfig()

        resolved_transform = (
            transform
            if transform is not None
            else AffineTransform.identity(ndim=len(data.level_shapes[0]))
        )
        visual_model = MultiscaleImageVisual(
            name=name,
            data_store_id=str(data.id),
            level_transforms=data.level_transforms,
            appearance=appearance,
            render_config=render_config,
            transform=resolved_transform,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

    def add_multichannel_image(
        self,
        data: ImageMemoryStore,
        scene_id: UUID,
        channel_axis: int,
        channels: dict[int, ChannelAppearance],
        name: str = "multichannel_image",
        max_channels_2d: int = 8,
        max_channels_3d: int = 4,
    ) -> MultichannelImageVisual:
        """Add an in-memory multichannel image visual to a scene.

        Parameters
        ----------
        data : ImageMemoryStore
            Backing data store.
        scene_id : UUID
            Target scene.
        channel_axis : int
            Data axis index for the channel dimension.
        channels : dict[int, ChannelAppearance]
            Per-channel appearance keyed by channel index.
        name : str
            Display name for the visual.
        max_channels_2d : int
            Maximum simultaneous 2D channel nodes.
        max_channels_3d : int
            Maximum simultaneous 3D channel nodes.

        Returns
        -------
        MultichannelImageVisual
        """
        if len(channels) > max_channels_2d:
            raise ValueError(
                f"len(channels)={len(channels)} exceeds "
                f"max_channels_2d={max_channels_2d}."
            )
        visual_model = MultichannelImageVisual(
            name=name,
            data_store_id=str(data.id),
            channel_axis=channel_axis,
            channels=channels,
            max_channels_2d=max_channels_2d,
            max_channels_3d=max_channels_3d,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

    def add_multichannel_image_multiscale(
        self,
        data: BaseDataStore,
        scene_id: UUID,
        channel_axis: int,
        channels: dict[int, ChannelAppearance],
        name: str = "multichannel_image",
        render_config: MultiscaleImageRenderConfig | None = None,
        transform: AffineTransform | None = None,
        max_channels_2d: int = 8,
        max_channels_3d: int = 4,
    ) -> MultichannelMultiscaleImageVisual:
        """Add a multiscale multichannel image visual to a scene.

        Parameters
        ----------
        data : BaseDataStore
            Backing multiscale data store.
        scene_id : UUID
            Target scene.
        channel_axis : int
            Data axis index for the channel dimension.
        channels : dict[int, ChannelAppearance]
            Per-channel appearance keyed by channel index.
        name : str
            Display name for the visual.
        render_config : MultiscaleImageRenderConfig or None
            LOD and rendering configuration; uses defaults when ``None``.
        transform : AffineTransform or None
            Data-to-world transform; uses identity when ``None``.
        max_channels_2d : int
            Maximum simultaneous 2D channel nodes.
        max_channels_3d : int
            Maximum simultaneous 3D channel nodes.

        Returns
        -------
        MultichannelMultiscaleImageVisual
        """
        if render_config is None:
            render_config = MultiscaleImageRenderConfig()
        if len(channels) > max_channels_2d:
            raise ValueError(
                f"len(channels)={len(channels)} exceeds "
                f"max_channels_2d={max_channels_2d}."
            )
        visual_model = MultichannelMultiscaleImageVisual(
            name=name,
            data_store_id=str(data.id),
            channel_axis=channel_axis,
            channels=channels,
            level_transforms=data.level_transforms,
            render_config=render_config,
            max_channels_2d=max_channels_2d,
            max_channels_3d=max_channels_3d,
        )
        return self.add_visual(scene_id, visual_model, data_store=data)

    def _get_visual_model(self, visual_id: UUID) -> VisualType:
        """Return the visual model for *visual_id*.

        Raises
        ------
        KeyError
            If *visual_id* is not registered.
        """
        scene_id = self._visual_to_scene[visual_id]
        scene = self._model.scenes[scene_id]
        for visual in scene.visuals:
            if visual.id == visual_id:
                return visual
        raise KeyError(f"Visual {visual_id} not found.")

    def add_channel(
        self,
        visual_id: UUID,
        channel_index: int,
        appearance: ChannelAppearance,
    ) -> None:
        """Add a channel to a multichannel image visual.

        Parameters
        ----------
        visual_id : UUID
            ID of a MultichannelImageVisual or MultichannelMultiscaleImageVisual.
        channel_index : int
            Index along the visual's channel_axis. Must not already be present.
        appearance : ChannelAppearance
            Colormap, clim, and opacity settings for the new channel.

        Raises
        ------
        ValueError
            If channel_index is already present.
        RuntimeError
            If the pool is full.
        """
        visual = self._get_visual_model(visual_id)
        if channel_index in visual.channels:
            raise ValueError(
                f"channel_index={channel_index} already in visual.channels."
            )
        if len(visual.channels) >= visual.max_channels_2d:
            raise RuntimeError(
                f"Pool is full ({visual.max_channels_2d} channels). "
                "Increase max_channels_2d or remove a channel first."
            )
        new_channels = dict(visual.channels)
        new_channels[channel_index] = appearance
        visual.channels = new_channels
        self.reslice_visual(visual_id)

    def remove_channel(self, visual_id: UUID, channel_index: int) -> None:
        """Remove a channel from a multichannel image visual.

        Parameters
        ----------
        visual_id : UUID
            ID of a MultichannelImageVisual or MultichannelMultiscaleImageVisual.
        channel_index : int
            Index of the channel to remove.

        Raises
        ------
        KeyError
            If channel_index is not in visual.channels.
        """
        visual = self._get_visual_model(visual_id)
        if channel_index not in visual.channels:
            raise KeyError(f"channel_index={channel_index} not in visual.channels.")
        new_channels = {k: v for k, v in visual.channels.items() if k != channel_index}
        visual.channels = new_channels

    # ------------------------------------------------------------------
    # Visual management — private dispatch methods
    # ------------------------------------------------------------------

    def _add_multiscale_image_visual(
        self,
        scene_id: UUID,
        visual_model: MultiscaleImageVisual,
    ) -> MultiscaleImageVisual:
        """Wire and register a pre-built MultiscaleImageVisual.

        Parameters
        ----------
        scene_id : UUID
            ID of an existing scene.
        visual_model : MultiscaleImageVisual
            Pre-built visual model. Its ``data_store_id`` must already be
            registered in ``self._model.data.stores``.

        Returns
        -------
        MultiscaleImageVisual
        """
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )

        scene.visuals.append(visual_model)

        level_shapes = list(data_store.level_shapes)
        gfx_visual = GFXMultiscaleImageVisual.from_cellier_model(
            model=visual_model,
            level_shapes=level_shapes,
            render_modes=render_modes,
            displayed_axes=displayed_axes,
        )

        self._render_manager.add_visual(
            scene_id, gfx_visual, data_store, displayed_axes
        )
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

    def _add_image_visual(
        self,
        scene_id: UUID,
        visual_model: ImageVisual,
    ) -> ImageVisual:
        """Wire and register a pre-built ImageVisual.

        Parameters
        ----------
        scene_id : UUID
            ID of an existing scene.
        visual_model : ImageVisual
            Pre-built visual model.

        Returns
        -------
        ImageVisual
        """
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )

        scene.visuals.append(visual_model)

        gfx_visual = GFXImageMemoryVisual(
            visual_model=visual_model,
            data_store=data_store,
            render_modes=render_modes,
        )

        self._render_manager.add_visual(
            scene_id, gfx_visual, data_store, displayed_axes
        )
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

    def _add_multichannel_image_memory_visual(
        self,
        scene_id: UUID,
        visual_model: MultichannelImageVisual,
    ) -> MultichannelImageVisual:
        """Wire and register a pre-built MultichannelImageVisual."""
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )

        scene.visuals.append(visual_model)

        gfx_visual = GFXMultichannelImageMemoryVisual(
            visual_model=visual_model,
            data_store=data_store,
            render_modes=render_modes,
        )

        self._render_manager.add_visual(
            scene_id, gfx_visual, data_store, displayed_axes
        )
        self._visual_to_scene[visual_model.id] = scene_id

        self._wire_aabb(visual_model)
        self._wire_transform(visual_model, scene_id)
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

    def _add_multichannel_multiscale_image_visual(
        self,
        scene_id: UUID,
        visual_model: MultichannelMultiscaleImageVisual,
    ) -> MultichannelMultiscaleImageVisual:
        """Wire and register a pre-built MultichannelMultiscaleImageVisual."""
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )

        scene.visuals.append(visual_model)

        level_shapes = list(data_store.level_shapes)
        gfx_visual = GFXMultichannelMultiscaleImageVisual(
            visual_model=visual_model,
            level_shapes=level_shapes,
            render_modes=render_modes,
            displayed_axes=displayed_axes,
        )

        self._render_manager.add_visual(
            scene_id, gfx_visual, data_store, displayed_axes
        )
        self._visual_to_scene[visual_model.id] = scene_id

        self._wire_aabb(visual_model)
        self._wire_transform(visual_model, scene_id)
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

    def _add_points_visual(
        self,
        scene_id: UUID,
        visual_model: PointsVisual,
    ) -> PointsVisual:
        """Wire and register a pre-built PointsVisual.

        Parameters
        ----------
        scene_id : UUID
            ID of an existing scene.
        visual_model : PointsVisual
            Pre-built visual model.

        Returns
        -------
        PointsVisual
        """
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )

        scene.visuals.append(visual_model)

        gfx_visual = GFXPointsMemoryVisual(
            visual_model=visual_model,
            render_modes=render_modes,
            transform=visual_model.transform,
        )

        self._render_manager.add_visual(
            scene_id, gfx_visual, data_store, displayed_axes
        )
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

    def _add_lines_visual(
        self,
        scene_id: UUID,
        visual_model: LinesVisual,
    ) -> LinesVisual:
        """Wire and register a pre-built LinesVisual.

        Parameters
        ----------
        scene_id : UUID
            ID of an existing scene.
        visual_model : LinesVisual
            Pre-built visual model.

        Returns
        -------
        LinesVisual
        """
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )

        scene.visuals.append(visual_model)

        gfx_visual = GFXLinesMemoryVisual(
            visual_model=visual_model,
            render_modes=render_modes,
            transform=visual_model.transform,
        )

        self._render_manager.add_visual(
            scene_id, gfx_visual, data_store, displayed_axes
        )
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

    def _add_mesh_visual(
        self,
        scene_id: UUID,
        visual_model: MeshVisual,
    ) -> MeshVisual:
        """Wire and register a pre-built MeshVisual.

        Parameters
        ----------
        scene_id : UUID
            ID of an existing scene.
        visual_model : MeshVisual
            Pre-built visual model.

        Returns
        -------
        MeshVisual
        """
        import warnings

        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]
        scene = self._model.scenes[scene_id]
        displayed_axes = scene.dims.selection.displayed_axes
        render_modes = self._scene_render_modes.get(
            scene_id, {"3d"} if len(displayed_axes) == 3 else {"2d"}
        )

        # Warn if Phong requested but no lights in scene.
        if isinstance(visual_model.appearance, MeshPhongAppearance):
            if not self._render_manager.scene_has_lighting(scene_id):
                warnings.warn(
                    "MeshPhongAppearance requires lights in the scene. "
                    "Pass lighting='default' to the Scene model, otherwise "
                    "the mesh will render black.",
                    stacklevel=3,
                )

        scene.visuals.append(visual_model)

        gfx_visual = GFXMeshMemoryVisual(
            visual_model=visual_model,
            render_modes=render_modes,
            transform=visual_model.transform,
        )

        self._render_manager.add_visual(
            scene_id, gfx_visual, data_store, displayed_axes
        )
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

    # ------------------------------------------------------------------
    # Overlay construction
    # ------------------------------------------------------------------

    def _build_gfx_overlay(
        self,
        canvas_id: UUID,
        overlay_model: CanvasOverlay,
    ):
        """Construct the render-layer overlay for *overlay_model*.

        Mirrors the ``_add_*_visual`` pattern: the controller is responsible
        for constructing GFX objects; the render manager is a passive registrar.

        Parameters
        ----------
        canvas_id : UUID
            ID of the canvas that will own the overlay.
        overlay_model : CanvasOverlay
            Model-layer overlay description.

        Raises
        ------
        TypeError
            If *overlay_model* has an unrecognised type.
        """
        canvas_view = self._render_manager._canvases[canvas_id]
        if isinstance(overlay_model, CenteredAxes2D):
            return GFXCenteredAxes2D(
                model=overlay_model,
                camera=canvas_view.camera,
            )
        raise TypeError(
            f"Unrecognised overlay type {type(overlay_model)!r}. "
            "Register a handler in _build_gfx_overlay."
        )

    # ------------------------------------------------------------------
    # Canvas management
    # ------------------------------------------------------------------

    def add_canvas(
        self,
        scene_id: UUID,
        render_modes: set[str] | None = None,
        initial_dim: str | None = None,
        fov: float = 70.0,
        depth_range_3d: tuple[float, float] = (1.0, 8000.0),
        depth_range_2d: tuple[float, float] = (-500.0, 500.0),
    ) -> QWidget:
        """Create a canvas attached to a scene and return its embeddable widget.

        Parameters
        ----------
        scene_id : UUID
            ID of an existing scene.
        render_modes : set[str] or None
            Which camera modes to prepare on the canvas.  Each entry must be
            ``"2d"`` or ``"3d"``.  When ``None``, defaults to the scene's own
            ``render_modes``.  Pass ``{"2d", "3d"}`` for a canvas that can
            switch between views.
        initial_dim : str or None
            Which mode is active when the canvas first appears.  Must be a
            member of ``render_modes``.  When ``None``, inferred from the
            scene's current ``displayed_axes`` length (3 axes -> ``"3d"``,
            otherwise ``"2d"``).
        fov : float
            Vertical field of view in degrees for the 3D perspective camera.
            Ignored when ``"3d"`` is not in ``render_modes``.  Default ``70.0``.
        depth_range_3d : tuple[float, float]
            ``(near, far)`` clip distances for the 3D perspective camera.
            Default ``(1.0, 8000.0)``.
        depth_range_2d : tuple[float, float]
            ``(near, far)`` clip distances for the 2D orthographic camera.
            Default ``(-500.0, 500.0)``.

        Returns
        -------
        QWidget
            The render widget.  Embed with ``layout.addWidget(widget)``.

        Raises
        ------
        ValueError
            If ``initial_dim`` is supplied but is not a member of
            ``render_modes``.
        """
        scene = self._model.scenes[scene_id]

        if render_modes is None:
            render_modes = scene.render_modes

        if initial_dim is None:
            initial_dim = (
                "3d" if len(scene.dims.selection.displayed_axes) == 3 else "2d"
            )

        if initial_dim not in render_modes:
            raise ValueError(
                f"initial_dim={initial_dim!r} is not in render_modes={render_modes!r}."
            )

        cameras: dict[str, CameraType] = {}
        if "3d" in render_modes:
            cameras["3d"] = PerspectiveCamera(
                fov=fov,
                near_clipping_plane=depth_range_3d[0],
                far_clipping_plane=depth_range_3d[1],
                controller=OrbitCameraController(enabled=True),
            )
        if "2d" in render_modes:
            cameras["2d"] = OrthographicCamera(
                near_clipping_plane=depth_range_2d[0],
                far_clipping_plane=depth_range_2d[1],
                controller=PanZoomCameraController(enabled=True),
            )

        canvas_model = Canvas(cameras=cameras)
        return self.add_canvas_model(scene_id, canvas_model, initial_dim=initial_dim)

    def add_canvas_model(
        self,
        scene_id: UUID,
        canvas_model: Canvas,
        initial_dim: str | None = None,
    ) -> QWidget:
        """Register a pre-built Canvas model with a scene.

        Used by ``from_model`` to restore canvases from a serialized
        ViewerModel, and called internally by ``add_canvas``.  Camera state
        (position, rotation, fov, depth range) is read from the camera models
        stored in ``canvas_model.cameras``.

        Parameters
        ----------
        scene_id : UUID
            ID of an existing scene.
        canvas_model : Canvas
            Pre-built canvas model.  Must have at least one entry in
            ``canvas_model.cameras``.
        initial_dim : str or None
            Which dim to activate first.  When ``None``, the first key in
            ``canvas_model.cameras`` is used (insertion order is preserved by
            Python dicts, so this is deterministic for serialized models).

        Returns
        -------
        QWidget
            The render widget.

        Raises
        ------
        ValueError
            If ``canvas_model.cameras`` is empty, or if ``initial_dim`` is not
            a key in ``canvas_model.cameras``.
        """
        if not canvas_model.cameras:
            raise ValueError("canvas_model.cameras must not be empty.")

        if initial_dim is None:
            initial_dim = next(iter(canvas_model.cameras))

        if initial_dim not in canvas_model.cameras:
            raise ValueError(
                f"initial_dim={initial_dim!r} is not a key in "
                f"canvas_model.cameras ({list(canvas_model.cameras)!r})."
            )

        active_camera = canvas_model.cameras[initial_dim]
        depth_range = (
            active_camera.near_clipping_plane,
            active_camera.far_clipping_plane,
        )

        # Store the model using its existing id.
        self._model.scenes[scene_id].canvases[canvas_model.id] = canvas_model

        # Use canvas_model.id as the single canonical ID for both layers.
        if isinstance(active_camera, PerspectiveCamera):
            canvas_view = self._render_manager.add_canvas(
                canvas_model.id,
                scene_id,
                parent=self._widget_parent,
                dim=initial_dim,
                fov=active_camera.fov,
                depth_range=depth_range,
            )
        else:
            # OrthographicCamera: fov is not meaningful; omit it so CanvasView
            # uses its own internal default for the 3D camera it keeps in reserve.
            canvas_view = self._render_manager.add_canvas(
                canvas_model.id,
                scene_id,
                parent=self._widget_parent,
                dim=initial_dim,
                depth_range=depth_range,
            )

        canvas_view.set_event_bus(self._event_bus)

        # Wire any overlays already stored on this canvas model to the render
        # layer.  For canvases created via add_canvas() this loop is a no-op
        # (overlays=[]).  For canvases restored from a serialized ViewerModel
        # the overlays list is already populated, so this call is sufficient —
        # from_model needs no additional overlay-restoration step.
        # _build_gfx_overlay is called directly rather than add_canvas_overlay_model
        # to avoid re-appending models that are already in canvas_model.overlays.
        for overlay_model in canvas_model.overlays:
            gfx_overlay = self._build_gfx_overlay(canvas_model.id, overlay_model)
            self._render_manager.add_canvas_overlay(canvas_model.id, gfx_overlay)

        self._canvas_to_scene[canvas_model.id] = scene_id
        self._scene_to_canvases[scene_id].append(canvas_model.id)

        return canvas_view.widget

    # ------------------------------------------------------------------
    # Scene and visual lookup
    # ------------------------------------------------------------------

    def get_scene(self, scene_id: UUID) -> Scene:
        """Return the live Scene model for scene_id."""
        return self._model.scenes[scene_id]

    def fit_camera(self, scene_id: UUID, canvas_id: UUID | None = None) -> None:
        """Fit the camera to the current scene bounding box.

        Safe to call immediately after ``add_image`` / ``add_image_multiscale``
        and transform assignment — the node matrix is set at construction time
        so no chunk data needs to be loaded first.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene whose camera should be fitted.
        canvas_id : UUID or None
            If provided, fit only that canvas.  When ``None`` (default),
            all canvases attached to *scene_id* are fitted.
        """
        gfx_scene = self._render_manager.get_scene(scene_id)
        if canvas_id is not None:
            canvas = self._render_manager._canvases.get(canvas_id)
            if canvas is not None:
                canvas.show_object(gfx_scene)
        else:
            for cid in self._scene_to_canvases.get(scene_id, []):
                canvas = self._render_manager._canvases.get(cid)
                if canvas is not None:
                    canvas.show_object(gfx_scene)

    def add_canvas_overlay_model(
        self,
        canvas_id: UUID,
        overlay: CanvasOverlay,
    ) -> CanvasOverlay:
        """Attach a screen-space overlay to a specific canvas.

        The overlay is rendered as a post-pass on top of the main scene each
        frame.  It does not participate in reslicing, has no world-space
        transform, and is not added to ``scene.visuals``.

        The overlay model is stored in the ``Canvas.overlays`` list of
        *canvas_id*, making it part of the serializable model.

        Parameters
        ----------
        canvas_id : UUID
            ID of the canvas that should display the overlay.  Use
            :meth:`get_canvas_ids` to look up canvas IDs for a scene.
        overlay : CanvasOverlay
            Model-layer overlay description.  Typically a
            :class:`~cellier.v2.visuals._canvas_overlay.CenteredAxes2D`.

        Returns
        -------
        CanvasOverlay
            The same overlay object passed in (for ID access or chaining).

        Raises
        ------
        KeyError
            If *canvas_id* is not registered.
        """
        scene_id = self._canvas_to_scene[canvas_id]
        canvas_model = self._model.scenes[scene_id].canvases[canvas_id]
        canvas_model.overlays.append(overlay)

        gfx_overlay = self._build_gfx_overlay(canvas_id, overlay)
        self._render_manager.add_canvas_overlay(canvas_id, gfx_overlay)

        return overlay

    def set_overlay_visible(self, overlay_id: UUID, visible: bool) -> None:
        """Toggle the visibility of a canvas overlay.

        Searches all canvases across all scenes for an overlay with
        ``overlay_id``.  Updates both the model field and the render layer.

        Parameters
        ----------
        overlay_id : UUID
            ID of the :class:`~cellier.v2.visuals._canvas_overlay.CanvasOverlay`
            to toggle.
        visible : bool
            ``True`` to show the overlay, ``False`` to hide it.

        Raises
        ------
        KeyError
            If no overlay with ``overlay_id`` is found.
        """
        for scene in self._model.scenes.values():
            for canvas_model in scene.canvases.values():
                for overlay_model in canvas_model.overlays:
                    if overlay_model.id == overlay_id:
                        overlay_model.visible = visible
                        canvas_view = self._render_manager._canvases.get(
                            canvas_model.id
                        )
                        if canvas_view is not None:
                            for gfx_overlay in canvas_view._overlays:
                                model_ref = getattr(gfx_overlay, "_model", None)
                                if model_ref is overlay_model:
                                    gfx_overlay.set_visible(visible)
                        return
        raise KeyError(
            f"No canvas overlay with id={overlay_id!r} found.  "
            "Ensure add_canvas_overlay was called before set_overlay_visible."
        )

    def get_scene_by_name(self, name: str) -> Scene:
        """Return the live Scene model for the given name.

        Raises KeyError if no scene with that name exists.
        """
        for scene in self._model.scenes.values():
            if scene.name == name:
                return scene
        raise KeyError(f"No scene named {name!r}")

    def get_canvas_ids(self, scene_id: UUID) -> list[UUID]:
        """Return the IDs of all canvases registered for *scene_id*.

        Parameters
        ----------
        scene_id : UUID
            ID of an existing scene.

        Returns
        -------
        list[UUID]
            Canvas IDs in registration order.  Empty if no canvases have
            been added yet.
        """
        return list(self._scene_to_canvases.get(scene_id, []))

    def get_canvas_view(self, canvas_id: UUID) -> CanvasView:
        """Return the render-layer ``CanvasView`` for *canvas_id*.

        Provides access to the rendering backend (camera, widget, overlays)
        for a canvas registered via :meth:`add_canvas` or
        :meth:`add_canvas_model`.

        Parameters
        ----------
        canvas_id : UUID
            ID of a registered canvas.

        Returns
        -------
        CanvasView

        Raises
        ------
        KeyError
            If *canvas_id* is not registered.
        """
        return self._render_manager._canvases[canvas_id]

    def get_camera_state(self, canvas_id: UUID) -> CameraState:
        """Return a snapshot of the current camera state for *canvas_id*.

        Useful for seeding downstream widgets (e.g. orientation overlays)
        with the post-fit camera state without constructing a synthetic
        ``CameraChangedEvent``.

        Parameters
        ----------
        canvas_id : UUID
            ID of a registered canvas.

        Returns
        -------
        CameraState

        Raises
        ------
        KeyError
            If *canvas_id* is not registered.
        """
        return self.get_canvas_view(canvas_id).capture_camera_state()

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
            resolved_source_id = _source_id_override.get() or self._id
            _SOURCE_ID_LOGGER.debug(
                "bridge  handler=_on_dims_psygnal  scene=%s"
                "  resolved_source=%s  override_active=%s",
                scene_id,
                resolved_source_id,
                _source_id_override.get() is not None,
            )
            self._event_bus.emit(
                DimsChangedEvent(
                    source_id=resolved_source_id,
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
        handler = self._make_transform_handler(visual.id, scene_id)
        visual.events.transform.connect(handler)
        self._visual_psygnal_handlers.setdefault(visual.id, []).append(
            (visual.events.transform, handler)
        )

    def _make_transform_handler(self, visual_id: UUID, scene_id: UUID) -> Callable:
        """Return a handler that emits TransformChangedEvent and triggers reslice.

        The reslice is skipped when ``_suppress_reslice`` is True, which is
        managed by the :meth:`suppress_reslice` context manager.
        """

        def _on_transform(new_transform: AffineTransform) -> None:
            self._event_bus.emit(
                TransformChangedEvent(
                    source_id=self._id,
                    scene_id=scene_id,
                    visual_id=visual_id,
                    transform=new_transform,
                )
            )
            if not self._suppress_reslice:
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
        handler = self._make_appearance_handler(visual.id)
        visual.appearance.events.connect(handler)
        self._visual_psygnal_handlers.setdefault(visual.id, []).append(
            (visual.appearance.events, handler)
        )

    def _wire_aabb(self, visual: MultiscaleImageVisual | ImageVisual) -> None:
        """Subscribe to all field changes on a visual's aabb model."""
        handler = self._make_aabb_handler(visual.id)
        visual.aabb.events.connect(handler)
        self._visual_psygnal_handlers.setdefault(visual.id, []).append(
            (visual.aabb.events, handler)
        )

    def _make_aabb_handler(self, visual_id: UUID) -> Callable:
        """Return a psygnal catch-all handler for a visual's AABBParams."""

        def _on_aabb_psygnal(info: EmissionInfo) -> None:
            field_name: str = info.signal.name
            new_value = info.args[0]
            resolved_source_id = _aabb_source_id_override.get() or self._id
            _SOURCE_ID_LOGGER.debug(
                "bridge  handler=_on_aabb_psygnal  visual=%s  field=%s"
                "  resolved_source=%s  override_active=%s",
                visual_id,
                field_name,
                resolved_source_id,
                _aabb_source_id_override.get() is not None,
            )
            self._event_bus.emit(
                AABBChangedEvent(
                    source_id=resolved_source_id,
                    visual_id=visual_id,
                    field_name=field_name,
                    new_value=new_value,
                )
            )

        return _on_aabb_psygnal

    def _make_appearance_handler(self, visual_id: UUID) -> Callable:
        """Return a psygnal catch-all handler for any visual's appearance model.

        Routes field changes to one of two bus events: ``visible`` field changes
        become ``VisualVisibilityChangedEvent``; all other fields become
        ``AppearanceChangedEvent`` with ``requires_reslice=True`` for fields in
        ``_RESLICE_FIELDS`` (``lod_bias``, ``force_level``, ``frustum_cull``).
        """

        def _on_appearance_psygnal(info: EmissionInfo) -> None:
            field_name: str = info.signal.name
            new_value = info.args[0]
            resolved_source_id = _source_id_override.get() or self._id
            _SOURCE_ID_LOGGER.debug(
                "bridge  handler=_on_appearance_psygnal  visual=%s  field=%s"
                "  resolved_source=%s  override_active=%s",
                visual_id,
                field_name,
                resolved_source_id,
                _source_id_override.get() is not None,
            )
            if field_name == "visible":
                self._event_bus.emit(
                    VisualVisibilityChangedEvent(
                        source_id=resolved_source_id,
                        visual_id=visual_id,
                        visible=new_value,
                    )
                )
            else:
                self._event_bus.emit(
                    AppearanceChangedEvent(
                        source_id=resolved_source_id,
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

    @contextmanager
    def suppress_reslice(self) -> Generator[None, None, None]:
        """Context manager that blocks reslice_scene inside transform handlers.

        Use this when updating a visual's transform without needing to reload
        its underlying data — for example, repositioning a static-geometry mesh
        by translation only.

        .. warning::
            ``_suppress_reslice`` is a flat boolean.  Nested calls or concurrent
            async tasks that mutate transforms inside overlapping
            ``suppress_reslice`` blocks will interfere.  Replace with a depth
            counter if that becomes necessary.
        """
        self._suppress_reslice = True
        try:
            yield
        finally:
            self._suppress_reslice = False

    def set_visual_transform(
        self,
        visual_id: UUID,
        transform: AffineTransform,
        *,
        reslice: bool = True,
    ) -> None:
        """Update the data-to-world transform of a visual.

        Assigns *transform* to the live visual model, which fires its psygnal
        field event and propagates the change to the render layer via the
        event bus.

        Parameters
        ----------
        visual_id : UUID
            ID of the visual to update.
        transform : AffineTransform
            New data-to-world transform.
        reslice : bool
            If True (default), a full reslice is triggered after the transform
            is applied — required for image visuals where the transform changes
            which data falls in the current slab.  Pass False for
            static-geometry visuals (mesh, points, lines) where the transform
            only repositions the node and the underlying data is unchanged.
        """
        visual_model = self.get_visual_model(visual_id)
        if reslice:
            visual_model.transform = transform
        else:
            with self.suppress_reslice():
                visual_model.transform = transform

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
        resolved_source_id = source_id if source_id is not None else self._id
        _SOURCE_ID_LOGGER.debug(
            "set  scene=%s  source=%s",
            scene_id,
            resolved_source_id,
        )
        token = _source_id_override.set(source_id)
        try:
            self._model.scenes[scene_id].dims.selection.slice_indices = slice_indices
        finally:
            _source_id_override.reset(token)
            _SOURCE_ID_LOGGER.debug("reset  scene=%s", scene_id)

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
        resolved_source_id = source_id if source_id is not None else self._id
        _SOURCE_ID_LOGGER.debug(
            "set  field=%s  visual=%s  source=%s",
            field,
            visual_id,
            resolved_source_id,
        )
        token = _source_id_override.set(source_id)
        try:
            setattr(visual.appearance, field, value)
        finally:
            _source_id_override.reset(token)
            _SOURCE_ID_LOGGER.debug("reset  field=%s  visual=%s", field, visual_id)

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
        resolved_source_id = source_id if source_id is not None else self._id
        _SOURCE_ID_LOGGER.debug(
            "set  field=%s  visual=%s  source=%s",
            field,
            visual_id,
            resolved_source_id,
        )
        token = _aabb_source_id_override.set(source_id)
        try:
            setattr(visual.aabb, field, value)
        finally:
            _aabb_source_id_override.reset(token)
            _SOURCE_ID_LOGGER.debug("reset  field=%s  visual=%s", field, visual_id)

    def update_displayed_axes(
        self,
        scene_id: UUID,
        displayed_axes: tuple[int, ...],
        *,
        source_id: UUID | None = None,
    ) -> None:
        """Set ``displayed_axes`` on a scene's dims.

        Tags the emitted bus event with *source_id*.
        GUI widgets should pass ``source_id=self._id`` so their own
        ``DimsChangedEvent`` subscription can ignore the echo.

        Parameters
        ----------
        scene_id :
            Target scene.
        displayed_axes :
            Tuple of axis indices to display; length 2 for 2D, 3 for 3D.
        source_id :
            UUID to stamp on the emitted ``DimsChangedEvent``.  Defaults
            to the controller's own ID.
        """
        token = _source_id_override.set(source_id)
        try:
            self._model.scenes[scene_id].dims.selection.displayed_axes = displayed_axes
        finally:
            _source_id_override.reset(token)

    # ------------------------------------------------------------------
    # IncomingEventBus handlers
    # ------------------------------------------------------------------

    def _on_appearance_update(self, event: AppearanceUpdateEvent) -> None:
        self.update_appearance_field(
            event.visual_id, event.field, event.value, source_id=event.source_id
        )

    def _on_dims_update(self, event: DimsUpdateEvent) -> None:
        if event.slice_indices is not None:
            self.update_slice_indices(
                event.scene_id, event.slice_indices, source_id=event.source_id
            )
        if event.displayed_axes is not None:
            self.update_displayed_axes(
                event.scene_id, event.displayed_axes, source_id=event.source_id
            )

    def _on_aabb_update(self, event: AABBUpdateEvent) -> None:
        self.update_aabb_field(
            event.visual_id, event.field, event.value, source_id=event.source_id
        )

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
        """Synchronous bus handler: updates camera model and schedules settle task."""
        self._update_camera_model(event.scene_id, event.source_id, event.camera_state)

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

    def _update_camera_model(
        self, scene_id: UUID, canvas_id: UUID, camera_state: CameraState
    ) -> None:
        """Write a CameraState snapshot back into the model-layer camera.

        Branches on the actual model type: ``PerspectiveCamera`` writes
        ``up_direction`` and ``fov``; ``OrthographicCamera`` writes ``width``
        and ``height`` (``extent``).

        Parameters
        ----------
        scene_id : UUID
            Scene that owns the canvas.
        canvas_id : UUID
            The specific canvas whose camera moved (``event.source_id``).
        camera_state : CameraState
            Snapshot to write back.
        """
        scene = self._model.scenes[scene_id]
        canvas_model = scene.canvases.get(canvas_id)
        if canvas_model is None:
            return
        canvas_view = self._render_manager._canvases.get(canvas_id)
        if canvas_view is not None:
            active_dim = canvas_view._dim
        else:
            # No render-layer canvas (e.g. headless tests): infer from type.
            active_dim = "2d" if camera_state.camera_type == "orthographic" else "3d"
        camera_model = canvas_model.cameras.get(active_dim)
        if camera_model is None:
            return

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
        canvas_id: UUID,
        view_direction: tuple[float, float, float] = (-1, -1, -1),
        up: tuple[float, float, float] = (0, 0, 1),
    ) -> None:
        """Fit the camera to a visual's bounding box.

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
        self._render_manager.look_at_visual(visual_id, canvas_id, view_direction, up)

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
        self._render_manager.set_camera_depth_range(canvas_id, depth_range)

    # ------------------------------------------------------------------
    # Stubs for future features
    # ------------------------------------------------------------------

    def add_labels(self, *args, **kwargs):
        """Not implemented in Phase 1."""
        raise NotImplementedError("add_labels is not implemented in Phase 1.")

    def add_paint_controller(
        self,
        visual_id: UUID,
        canvas_id: UUID,
        brush_value: float = 1.0,
        brush_radius_voxels: float = 2.0,
        history_depth: int = 100,
        autosave_interval_s: float | None = None,
        downsample_mode: str = "decimate",
    ):
        """Create and wire a paint controller for the visual's data store.

        Parameters
        ----------
        visual_id : UUID
            Visual to paint on.
        canvas_id : UUID
            Canvas to bind to.  Its camera controller is disabled for
            the session.  Pass ``controller.get_canvas_ids(scene_id)[0]``
            for the common single-canvas case.
        brush_value : float
            Scalar written to every painted voxel.
        brush_radius_voxels : float
            Brush radius in level-0 voxel units.
        history_depth : int
            Maximum undoable strokes.
        autosave_interval_s : float | None
            Seconds between automatic flushes for ``MultiscalePaintController``.
            Each autosave rebuilds the pyramid and resets GPU paint textures.
            ``None`` disables autosave.  Ignored for ``SyncPaintController``.
        downsample_mode : str
            ``"decimate"`` (default) for label stores; ``"mean"`` for
            intensity images.  Ignored for ``SyncPaintController``.

        Returns
        -------
        AbstractPaintController
            Fully wired; caller owns the object.

        Raises
        ------
        TypeError
            If the data store type has no registered paint controller.
        """
        from cellier.v2.data.image._image_memory_store import ImageMemoryStore
        from cellier.v2.data.image._ome_zarr_image_store import OMEZarrImageDataStore
        from cellier.v2.data.image._zarr_multiscale_store import (
            MultiscaleZarrDataStore,
        )

        scene_id = self._visual_to_scene[visual_id]
        scene = self._model.scenes[scene_id]
        visual_model = next(v for v in scene.visuals if v.id == visual_id)
        data_store = self._model.data.stores[UUID(visual_model.data_store_id)]

        if isinstance(data_store, ImageMemoryStore):
            from cellier.v2.paint import SyncPaintController

            displayed_axes = scene.dims.selection.displayed_axes
            if len(displayed_axes) != 2:
                raise NotImplementedError(
                    "SyncPaintController currently only supports 2-D "
                    f"displayed-axis configurations; scene {scene_id} has "
                    f"displayed_axes={displayed_axes!r}."
                )
            return SyncPaintController(
                cellier_controller=self,
                visual_id=visual_id,
                scene_id=scene_id,
                canvas_id=canvas_id,
                data_store=data_store,
                displayed_axes=displayed_axes,
                brush_value=brush_value,
                brush_radius_voxels=brush_radius_voxels,
                history_depth=history_depth,
            )

        if isinstance(data_store, (OMEZarrImageDataStore, MultiscaleZarrDataStore)):
            from cellier.v2.paint import MultiscalePaintController

            displayed_axes = scene.dims.selection.displayed_axes
            if len(displayed_axes) != 2:
                raise NotImplementedError(
                    "MultiscalePaintController currently only supports 2-D "
                    f"displayed-axis configurations; scene {scene_id} has "
                    f"displayed_axes={displayed_axes!r}.  3-D paint "
                    "feedback is a Phase 3 follow-up."
                )

            block_size = visual_model.render_config.block_size
            return MultiscalePaintController(
                cellier_controller=self,
                visual_id=visual_id,
                scene_id=scene_id,
                canvas_id=canvas_id,
                data_store=data_store,
                visual_block_size=block_size,
                displayed_axes=displayed_axes,
                brush_value=brush_value,
                brush_radius_voxels=brush_radius_voxels,
                history_depth=history_depth,
                autosave_interval_s=autosave_interval_s,
                downsample_mode=downsample_mode,
            )

        raise TypeError(
            f"No PaintController implementation for data store type "
            f"{type(data_store).__name__!r}.  "
            f"Supported: ImageMemoryStore, OMEZarrImageDataStore, "
            f"MultiscaleZarrDataStore."
        )

    def remove_scene(self, scene_id: UUID) -> None:
        """Remove a scene and all its visuals and canvases.

        Teardown order mirrors ``remove_visual`` for each child visual, then
        cleans up the scene-level maps and render layer.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene to remove.

        Raises
        ------
        KeyError
            If ``scene_id`` is not registered.
        """
        scene = self._model.scenes[scene_id]

        # 1. Tear down all child visuals (psygnal + bus + render layer each).
        for visual_model in list(scene.visuals):
            self.remove_visual(visual_model.id)

        # 2. Cancel any pending camera-settle tasks for this scene's canvases.
        for canvas_id in self._scene_to_canvases.get(scene_id, []):
            task = self._settle_tasks.pop(canvas_id, None)
            if task is not None and not task.done():
                task.cancel()

        # 3. Bus cleanup for canvases and the scene itself.
        for canvas_id in self._scene_to_canvases.pop(scene_id, []):
            self._event_bus.unsubscribe_all(canvas_id)
            self._canvas_to_scene.pop(canvas_id, None)
        self._event_bus.unsubscribe_all(scene_id)

        # 4. Clean up controller-side scene maps.
        self._dims_cache.pop(scene_id, None)
        self._scene_render_modes.pop(scene_id, None)

        # 5. Remove from model layer.
        self._model.scenes.pop(scene_id)

        # 6. Render-layer teardown (drops gfx.Scene, canvas widgets, GPU refs).
        self._render_manager.remove_scene(scene_id)

        # 7. Notify external observers.
        self._event_bus.emit(SceneRemovedEvent(source_id=self._id, scene_id=scene_id))

    def remove_visual(self, visual_id: UUID) -> None:
        """Remove a visual from its scene, disconnecting all wiring.

        Teardown order:
        1. Psygnal bridge handlers disconnected first — prevents the
           bridge closures from firing during any subsequent model access.
        2. Bus subscriptions removed — prevents dangling GFX-layer handlers
           from receiving events after the node is gone from the scene graph.
        3. Render-layer removal — drops scene-graph node and GPU references.
        4. ``VisualRemovedEvent`` emitted for external observers.

        Parameters
        ----------
        visual_id : UUID
            ID of the visual to remove.

        Raises
        ------
        KeyError
            If ``visual_id`` is not registered.
        """
        scene_id = self._visual_to_scene[visual_id]
        scene = self._model.scenes[scene_id]
        visual_model = self.get_visual_model(visual_id)

        # 1. Remove from model layer.
        scene.visuals.remove(visual_model)

        # 2. Disconnect psygnal bridge handlers.
        for signal, handler in self._visual_psygnal_handlers.pop(visual_id, []):
            signal.disconnect(handler)

        # 3. Remove bus subscriptions for this visual's GFX handlers.
        self._event_bus.unsubscribe_all(visual_id)

        # 4. Remove from controller lookup maps.
        self._visual_to_scene.pop(visual_id)

        # 5. Render-layer teardown.
        self._render_manager.remove_visual(visual_id)

        # 6. Notify external observers.
        self._event_bus.emit(
            VisualRemovedEvent(
                source_id=self._id,
                scene_id=scene_id,
                visual_id=visual_id,
            )
        )

    def remove_data_store(self, data_store_id: UUID) -> None:
        """Remove a data store from the model.

        Raises ``ValueError`` if any live visual still references the store.
        Call ``remove_visual`` for each referencing visual first.

        Parameters
        ----------
        data_store_id : UUID
            ID of the data store to remove.

        Raises
        ------
        ValueError
            If one or more visuals still reference the store.  The error
            message names each visual so the caller can identify them.
        KeyError
            If ``data_store_id`` is not registered.
        """
        referencing = [
            v
            for scene in self._model.scenes.values()
            for v in scene.visuals
            if UUID(v.data_store_id) == data_store_id
        ]
        if referencing:
            names = ", ".join(f"{v.name!r} ({v.id})" for v in referencing)
            raise ValueError(
                f"Cannot remove data store {data_store_id}: "
                f"still referenced by visuals: {names}"
            )
        self._model.data.stores.pop(data_store_id)

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
        owner_id: UUID,
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
            UUID under which this subscription is registered.  Pass the
            caller's own UUID so ``unsubscribe_owner(owner_id)`` removes it
            during teardown.
        weak :
            If True, hold only a weak reference to *callback*.  Use for
            transient widgets that may be destroyed outside the controller's
            teardown path.  Cannot be used with lambdas.

        Returns
        -------
        SubscriptionHandle
            Pass to ``EventBus.unsubscribe()`` for individual removal.
        """
        return self._event_bus.subscribe(
            DimsChangedEvent,
            callback,
            entity_id=scene_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_camera_changed(
        self,
        scene_id: UUID,
        callback: Callable[[CameraChangedEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired whenever the camera for *scene_id* changes.

        The callback receives a ``CameraChangedEvent`` carrying the latest
        ``CameraState``, including ``extent`` (width, height) for
        ``OrthographicCamera`` scenes.

        Parameters
        ----------
        scene_id :
            The scene to watch.
        callback :
            Called with the ``CameraChangedEvent`` on each camera change.
        owner_id :
            UUID under which this subscription is registered.  Pass the
            caller's own UUID so ``unsubscribe_owner(owner_id)`` removes it
            during teardown.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._event_bus.subscribe(
            CameraChangedEvent,
            callback,
            entity_id=scene_id,
            owner_id=owner_id,
            weak=weak,
        )

    def _on_raw_pointer_event(self, event: _CanvasRawPointerEvent) -> None:
        """Embed the 2D camera position in N-dimensional world space.

        Reads displayed_axes and slice_indices from the scene model to fill
        the full world coordinate, then emits the public canvas mouse event.
        No render-layer access occurs here.
        """
        from cellier.v2.events._events import CanvasPickInfo

        scene_model = self._model.scenes[event.scene_id]
        dims = scene_model.dims
        displayed_axes = dims.selection.displayed_axes
        slice_indices = dims.selection.slice_indices
        axis_labels = dims.coordinate_system.axis_labels
        n_dims = len(axis_labels)

        pick_info = CanvasPickInfo(hit_visual_id=event.hit_visual_id)

        if event.camera_type == "2d":
            world_coord = np.empty(n_dims, dtype=np.float64)
            # Embed pygfx position_2d into world_coord verbatim:
            #   world_coord[displayed_axes[0]] = position_2d[0]  (pygfx-x)
            #   world_coord[displayed_axes[1]] = position_2d[1]  (pygfx-y)
            # Both image visuals render data[r, c] at pygfx (x=c, y=r), so
            # world_coord[ax_row] == pygfx_x (column) and
            # world_coord[ax_col] == pygfx_y (row) after this assignment.
            # Paint controllers swap ax_row ↔ ax_col to recover (row, col)
            # voxel order before calling imap_coordinates.
            for i, axis in enumerate(displayed_axes):
                world_coord[axis] = event.position_2d[i]  # type: ignore[index]
            for axis, idx in slice_indices.items():
                world_coord[axis] = float(idx)

            _CLS_2D = {
                "press": CanvasMousePress2DEvent,
                "move": CanvasMouseMove2DEvent,
                "release": CanvasMouseRelease2DEvent,
            }
            self._event_bus.emit(
                _CLS_2D[event.action](
                    source_id=event.canvas_id,
                    scene_id=event.scene_id,
                    world_coordinate=world_coord,
                    pick_info=pick_info,
                )
            )
        else:
            _CLS_3D = {
                "press": CanvasMousePress3DEvent,
                "move": CanvasMouseMove3DEvent,
                "release": CanvasMouseRelease3DEvent,
            }
            self._event_bus.emit(
                _CLS_3D[event.action](
                    source_id=event.canvas_id,
                    scene_id=event.scene_id,
                    ray=event.ray,
                    pick_info=pick_info,
                )
            )

    def on_mouse_press_2d(
        self,
        canvas_id: UUID,
        callback: Callable[[CanvasMousePress2DEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired on every pointer-down event on a 2D canvas.

        Parameters
        ----------
        canvas_id :
            The canvas to watch.
        callback :
            Called with the CanvasMousePress2DEvent on each press.
        owner_id :
            UUID under which this subscription is registered for bulk
            removal via unsubscribe_all(owner_id).
        weak :
            If True, hold only a weak reference to *callback*.
        """
        return self._event_bus.subscribe(
            CanvasMousePress2DEvent,
            callback,
            entity_id=canvas_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_mouse_move_2d(
        self,
        canvas_id: UUID,
        callback: Callable[[CanvasMouseMove2DEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired on every pointer-move event on a 2D canvas."""
        return self._event_bus.subscribe(
            CanvasMouseMove2DEvent,
            callback,
            entity_id=canvas_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_mouse_release_2d(
        self,
        canvas_id: UUID,
        callback: Callable[[CanvasMouseRelease2DEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired on every pointer-up event on a 2D canvas."""
        return self._event_bus.subscribe(
            CanvasMouseRelease2DEvent,
            callback,
            entity_id=canvas_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_mouse_press_3d(
        self,
        canvas_id: UUID,
        callback: Callable[[CanvasMousePress3DEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired on every pointer-down event on a 3D canvas.

        Parameters
        ----------
        canvas_id :
            The canvas to watch.
        callback :
            Called with the CanvasMousePress3DEvent on each press.
        owner_id :
            UUID under which this subscription is registered for bulk
            removal via unsubscribe_all(owner_id).
        weak :
            If True, hold only a weak reference to *callback*.
        """
        return self._event_bus.subscribe(
            CanvasMousePress3DEvent,
            callback,
            entity_id=canvas_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_mouse_move_3d(
        self,
        canvas_id: UUID,
        callback: Callable[[CanvasMouseMove3DEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired on every pointer-move event on a 3D canvas."""
        return self._event_bus.subscribe(
            CanvasMouseMove3DEvent,
            callback,
            entity_id=canvas_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_mouse_release_3d(
        self,
        canvas_id: UUID,
        callback: Callable[[CanvasMouseRelease3DEvent], None],
        *,
        owner_id: UUID,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register a callback fired on every pointer-up event on a 3D canvas."""
        return self._event_bus.subscribe(
            CanvasMouseRelease3DEvent,
            callback,
            entity_id=canvas_id,
            owner_id=owner_id,
            weak=weak,
        )

    def set_camera_controller_enabled(self, canvas_id: UUID, enabled: bool) -> None:
        """Enable or disable the camera controller for one canvas.

        Parameters
        ----------
        canvas_id : UUID
            The canvas whose controller state should change.
        enabled : bool
            False disables the controller (paint session active).
            True restores normal camera interaction (session ended).
        """
        self._render_manager._canvases[canvas_id].set_controller_enabled(enabled)

    def invalidate_painted_tiles_2d(
        self, visual_id: UUID, dirty_grid_coords: set[tuple[int, int]]
    ) -> int:
        """Evict the visible 2D-cache tiles for *visual_id* listed in dirty grid coords.

        Used by :class:`MultiscalePaintController` after staging level-0
        writes so the next reslice re-fetches through the open paint
        transaction.  No-op for visuals that don't own a 2D tile cache.

        Parameters
        ----------
        visual_id :
            The painted multiscale visual.
        dirty_grid_coords :
            Set of ``(gy, gx)`` tile-grid coordinates whose finest-level
            tiles should be evicted.

        Returns
        -------
        int
            Number of tiles evicted (0 if the visual has no 2D cache or
            no entries matched).
        """
        scene_id = self._visual_to_scene[visual_id]
        scene_manager = self._render_manager._scenes[scene_id]
        gfx_visual = scene_manager.get_visual(visual_id)
        if not hasattr(gfx_visual, "invalidate_painted_tiles_2d"):
            return 0
        return gfx_visual.invalidate_painted_tiles_2d(dirty_grid_coords)

    def patch_painted_tiles_2d(
        self,
        visual_id: UUID,
        voxel_indices: np.ndarray,
        values: np.ndarray,
        displayed_axes: tuple[int, int],
    ) -> int:
        """Write paint into the visual's GPU paint cache (Phase-2 fast path).

        Used by :class:`MultiscalePaintController._write_values` for
        sub-frame visible feedback in 2-D paint sessions.  Mirrors the
        architectural pattern of :meth:`invalidate_painted_tiles_2d`.

        Parameters
        ----------
        visual_id :
            The painted multiscale visual.
        voxel_indices :
            Shape ``(N, ndim)`` int64.  Level-0 voxel indices in
            data-array axis order.
        values :
            Shape ``(N,)`` float32.  Brush values.
        displayed_axes :
            ``(row_axis, col_axis)`` for the 2-D display.

        Returns
        -------
        int
            Number of tiles successfully patched.  Returns 0 if the visual
            has no 2-D paint resources or every tile was rejected by the
            slot manager.
        """
        scene_id = self._visual_to_scene[visual_id]
        scene_manager = self._render_manager._scenes[scene_id]
        gfx_visual = scene_manager.get_visual(visual_id)
        if not hasattr(gfx_visual, "patch_paint_texture"):
            return 0
        return gfx_visual.patch_paint_texture(voxel_indices, values, displayed_axes)

    def clear_painted_tiles_2d(self, visual_id: UUID) -> None:
        """Clear the GPU paint textures for *visual_id*.

        Called by :class:`MultiscalePaintController.commit` and ``abort`` at
        session end.
        """
        scene_id = self._visual_to_scene[visual_id]
        scene_manager = self._render_manager._scenes[scene_id]
        gfx_visual = scene_manager.get_visual(visual_id)
        if not hasattr(gfx_visual, "clear_paint_textures"):
            return
        gfx_visual.clear_paint_textures()

    def on_visual_changed(
        self,
        visual_id: UUID,
        callback: Callable[[AppearanceChangedEvent], None],
        *,
        owner_id: UUID,
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
            UUID under which this subscription is registered.  Pass the
            caller's own UUID so ``unsubscribe_owner(owner_id)`` removes it
            during teardown.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._event_bus.subscribe(
            AppearanceChangedEvent,
            callback,
            entity_id=visual_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_aabb_changed(
        self,
        visual_id: UUID,
        callback: Callable[[AABBChangedEvent], None],
        *,
        owner_id: UUID,
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
            UUID under which this subscription is registered.  Pass the
            caller's own UUID so ``unsubscribe_owner(owner_id)`` removes it
            during teardown.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._event_bus.subscribe(
            AABBChangedEvent,
            callback,
            entity_id=visual_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_reslice_started(
        self,
        scene_id: UUID,
        callback: Callable[[ResliceStartedEvent], None],
        *,
        owner_id: UUID,
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
            UUID under which this subscription is registered.  Pass the
            caller's own UUID so ``unsubscribe_owner(owner_id)`` removes it
            during teardown.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._event_bus.subscribe(
            ResliceStartedEvent,
            callback,
            entity_id=scene_id,
            owner_id=owner_id,
            weak=weak,
        )

    def on_reslice_completed(
        self,
        visual_id: UUID,
        callback: Callable[[ResliceCompletedEvent], None],
        *,
        owner_id: UUID,
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
            UUID under which this subscription is registered.  Pass the
            caller's own UUID so ``unsubscribe_owner(owner_id)`` removes it
            during teardown.
        weak :
            If True, hold only a weak reference to *callback*.

        Returns
        -------
        SubscriptionHandle
        """
        return self._event_bus.subscribe(
            ResliceCompletedEvent,
            callback,
            entity_id=visual_id,
            owner_id=owner_id,
            weak=weak,
        )

    def unsubscribe_owner(self, owner_id: UUID) -> None:
        """Remove all event subscriptions registered under *owner_id*.

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

    def connect_widget(
        self,
        widget: Any,
        *,
        subscription_specs: list[SubscriptionSpec] | None = None,
    ) -> None:
        """Wire a widget's psygnal signals to the bus and register subscriptions.

        Widgets declare their intent through two psygnal signals and an optional
        list of ``SubscriptionSpec`` objects, so they never import or hold a
        reference to ``CellierController``.

        The caller is responsible for constructing the widget and passing the
        specs — typically obtained from ``widget.subscription_specs()``.

        Parameters
        ----------
        widget :
            Any object exposing:

            * ``widget._id`` — a ``UUID`` identifying the widget.
            * ``widget.changed`` — a psygnal ``Signal`` that emits a
              ``CellierUpdateEventTypes`` instance when the user changes a
              value.  Connected to ``incoming_events.emit``.
            * ``widget.closed`` — a psygnal ``Signal`` (no arguments) emitted
              when the widget is closed.  Triggers
              ``unsubscribe_owner(widget._id)``.

        subscription_specs :
            Optional list of ``SubscriptionSpec`` entries describing which
            outgoing bus events the widget wants to receive.  Pass ``None``
            (or omit the argument) for pure-output widgets that do not need
            model-driven updates.
        """
        widget.changed.connect(self._incoming_events.emit)
        owner_id = widget._id
        widget.closed.connect(lambda: self.unsubscribe_owner(owner_id))
        for spec in subscription_specs or []:
            self._event_bus.subscribe(
                spec.event_type,
                spec.handler,
                entity_id=spec.entity_id,
                owner_id=owner_id,
                weak=spec.weak,
            )
