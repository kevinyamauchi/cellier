"""CellierController — single entry-point for building Cellier v2 viewers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from cellier.v2.render._requests import DimsState
from cellier.v2.render._scene_config import VisualRenderConfig
from cellier.v2.render.render_manager import RenderManager
from cellier.v2.render.visuals._image import GFXMultiscaleImageVisual
from cellier.v2.scene.cameras import OrbitCameraController, PerspectiveCamera
from cellier.v2.scene.canvas import Canvas
from cellier.v2.scene.dims import CoordinateSystem, DimsManager
from cellier.v2.scene.scene import Scene
from cellier.v2.viewer_model import DataManager, ViewerModel
from cellier.v2.visuals._image import MultiscaleImageVisual

if TYPE_CHECKING:
    import pathlib

    from PySide6.QtWidgets import QWidget

    from cellier.v2.data._base_data_store import BaseDataStore
    from cellier.v2.visuals._image import ImageAppearance


class CellierController:
    """The main class for constructing and controlling a cellier visualization.

    Wraps a ViewerModel (model layer) and a RenderManager (render layer).
    """

    def __init__(self, widget_parent: QWidget | None = None) -> None:
        self._widget_parent = widget_parent
        self._model = ViewerModel(data=DataManager())
        self._render_manager = RenderManager()
        # Reverse map: visual_model_id → scene_id (model layer mirror of render layer)
        self._visual_to_scene: dict[UUID, UUID] = {}
        # Reverse map: canvas_id → scene_id
        self._canvas_to_scene: dict[UUID, UUID] = {}
        # Forward map: scene_id → list[canvas_id]
        self._scene_to_canvases: dict[UUID, list[UUID]] = {}

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
        if dim == "3d":
            displayed_axes = (0, 1, 2)
            slice_indices: tuple[int, ...] = ()
        elif dim == "2d":
            displayed_axes = (0, 1)
            slice_indices = (0,)
        else:
            raise ValueError(f"dim must be '2d' or '3d', got {dim!r}")

        dims = DimsManager(
            coordinate_system=coordinate_system,
            displayed_axes=displayed_axes,
            slice_indices=slice_indices,
        )
        scene = Scene(name=name, dims=dims)
        self._model.scenes[scene.id] = scene
        self._render_manager.add_scene(scene.id, dim=dim)
        self._scene_to_canvases[scene.id] = []
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

    def add_image(
        self,
        data: BaseDataStore,
        scene_id: UUID,
        appearance: ImageAppearance,
        name: str,
        block_size: int = 32,
        gpu_budget_bytes: int = 1 * 1024**3,
        threshold: float = 0.2,
        interpolation: str = "linear",
    ) -> MultiscaleImageVisual:
        """Add a multiscale image visual to a scene.

        Registers the data store (if not already registered), creates the
        MultiscaleImageVisual model, builds the GFXMultiscaleImageVisual
        render object, and wires everything together atomically.

        Parameters
        ----------
        data : BaseDataStore
            A MultiscaleZarrDataStore (or any BaseDataStore subclass that
            exposes ``level_shapes`` and ``n_levels``).
        scene_id : UUID
            ID of an existing scene (returned by ``add_scene``).
        appearance : ImageAppearance
            Appearance parameters for the visual.
        name : str
            Human-readable name for the visual.
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
            The live model object.  Mutate ``visual.appearance`` fields
            directly; they are read at the next ``reslice_*`` call.
        """
        if data.id not in self._model.data.stores:
            self._model.data.stores[data.id] = data

        scene = self._model.scenes[scene_id]
        dim = scene.dims.displayed_axes  # (0,1,2) → 3d, (0,1) → 2d

        downscale_factors = [2**i for i in range(data.n_levels)]

        visual_model = MultiscaleImageVisual(
            name=name,
            data_store_id=str(data.id),
            downscale_factors=downscale_factors,
            appearance=appearance,
        )

        scene.visuals.append(visual_model)

        render_modes = {"3d"} if len(dim) == 3 else {"2d"}

        level_shapes = list(data.level_shapes)
        gfx_visual = GFXMultiscaleImageVisual.from_cellier_model(
            model=visual_model,
            level_shapes=level_shapes,
            render_modes=render_modes,
            block_size=block_size,
            gpu_budget_bytes=gpu_budget_bytes,
            threshold=threshold,
            interpolation=interpolation,
        )

        self._render_manager.add_visual(scene_id, gfx_visual, data)
        self._visual_to_scene[visual_model.id] = scene_id

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

        render_modes = {"3d"} if len(scene.dims.displayed_axes) == 3 else {"2d"}
        level_shapes = list(data_store.level_shapes)
        gfx_visual = GFXMultiscaleImageVisual.from_cellier_model(
            model=visual_model,
            level_shapes=level_shapes,
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
        available_dims: str = "3d",
        fov: float = 70.0,
        depth_range: tuple[float, float] = (1.0, 8000.0),
    ) -> QWidget:
        """Create a canvas attached to a scene and return its embeddable widget.

        Parameters
        ----------
        scene_id : UUID
            ID of an existing scene.
        available_dims : str
            ``"3d"`` or ``"2d"``.
        fov : float
            Vertical field of view in degrees.
        depth_range : tuple[float, float]
            ``(near, far)`` clip distances.

        Returns
        -------
        QWidget
            The render widget.  Embed with ``layout.addWidget(widget)``.
        """
        canvas_id = uuid4()

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
            fov=fov,
            depth_range=depth_range,
        )

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
        dims = self._model.scenes[scene_id].dims
        return DimsState(
            displayed_axes=dims.displayed_axes,
            slice_indices=dims.slice_indices,
        )

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
        """Not implemented in Phase 1."""
        raise NotImplementedError("remove_scene is not implemented in Phase 1.")

    def remove_visual(self, visual_id: UUID) -> None:
        """Not implemented in Phase 1."""
        raise NotImplementedError("remove_visual is not implemented in Phase 1.")

    def remove_data_store(self, data_store_id: UUID) -> None:
        """Not implemented in Phase 1."""
        raise NotImplementedError("remove_data_store is not implemented in Phase 1.")

    def get_dims_widget(self, scene_id: UUID, parent: QWidget | None = None):
        """Return a dims slider widget for a scene.

        Not implemented in Phase 1.
        """
        raise NotImplementedError("get_dims_widget is not implemented in Phase 1.")

    def on_dims_changed(self, scene_id: UUID, callback) -> None:
        """Register a callback to be called when dims change.

        Not implemented in Phase 1.
        """
        raise NotImplementedError("on_dims_changed is not implemented in Phase 1.")
