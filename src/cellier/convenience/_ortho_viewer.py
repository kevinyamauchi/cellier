"""Four-panel orthoviewer convenience class wrapping CellierController.

An :class:`OrthoViewer` manages a single controller with four pre-wired scenes
-- three orthogonal 2D slice panels (``xy``, ``xz``, ``yz``) and one 3D volume
panel (``vol``) -- all sharing the same world coordinate system.  Like
:class:`~cellier.convenience.Viewer` it launches empty; the ``add_*`` methods
register one data store and fan a visual out to every panel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar
from uuid import UUID, uuid4

from cellier.controller import CellierController
from cellier.scene.dims import (
    AxisAlignedSelection,
    CoordinateSystem,
    DimsManager,
)
from cellier.scene.scene import Scene

if TYPE_CHECKING:
    from pathlib import Path

    from cellier.convenience._kwarg_dicts import (
        ChannelAppearanceKwargs,
        InMemoryImageAppearanceKwargs,
        InMemoryLabelsAppearanceKwargs,
        LinesMemoryAppearanceKwargs,
        MeshFlatAppearanceKwargs,
        MeshPhongAppearanceKwargs,
        MultiscaleImageAppearanceKwargs,
        MultiscaleImageRenderConfigKwargs,
        MultiscaleLabelRenderConfigKwargs,
        MultiscaleLabelsAppearanceKwargs,
        PointsMarkerAppearanceKwargs,
    )
    from cellier.data._base_data_store import BaseDataStore
    from cellier.data.image._image_memory_store import ImageMemoryStore
    from cellier.data.label._label_memory_store import LabelMemoryStore
    from cellier.data.lines._lines_memory_store import LinesMemoryStore
    from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
    from cellier.data.points._points_memory_store import PointsMemoryStore
    from cellier.events import DimsChangedEvent
    from cellier.render._config import RenderManagerConfig
    from cellier.transform import AffineTransform
    from cellier.visuals._channel_appearance import ChannelAppearance
    from cellier.visuals._image import (
        MultichannelMultiscaleImageVisual,
        MultiscaleImageAppearance,
        MultiscaleImageRenderConfig,
        MultiscaleImageVisual,
    )
    from cellier.visuals._image_memory import BaseImageAppearance, ImageVisual
    from cellier.visuals._image_memory_multichannel import MultichannelImageVisual
    from cellier.visuals._label_memory import BaseLabelsAppearance, LabelMemoryVisual
    from cellier.visuals._labels import (
        MultiscaleLabelRenderConfig,
        MultiscaleLabelsAppearance,
        MultiscaleLabelVisual,
    )
    from cellier.visuals._lines_memory import LinesMemoryAppearance, LinesVisual
    from cellier.visuals._mesh_memory import MeshAppearance, MeshVisual
    from cellier.visuals._points_memory import PointsMarkerAppearance, PointsVisual

_T = TypeVar("_T", bound="BaseDataStore")

# Panel keys in display order. ``vol`` is the 3D panel; the rest are 2D slices.
_PANEL_KEYS: tuple[str, ...] = ("xy", "xz", "yz", "vol")


class _ExtraAxisSyncer:
    """Fans non-spatial (extra) axis slice changes across all four panels.

    The three spatial axes are displayed/sliced differently per panel and are
    never synced.  Every other axis (channel, time, ...) represents the same
    global coordinate in all panels, so a change on one panel is propagated to
    the others.  ``update_slice_indices`` is a full replacement, so each target
    scene's current ``slice_indices`` is read, patched, and written back.
    """

    def __init__(
        self,
        controller: CellierController,
        scenes_by_id: dict[UUID, Scene],
        extra_axes: set[int],
    ) -> None:
        self._id = uuid4()
        self._controller = controller
        self._scenes_by_id = scenes_by_id
        self._extra_axes = set(extra_axes)
        self._syncing = False
        self.enabled = True

    @property
    def owner_id(self) -> UUID:
        """UUID under which this syncer's subscriptions are registered."""
        return self._id

    def handle(self, event: DimsChangedEvent) -> None:
        """Propagate extra-axis positions from the source scene to the others."""
        if not self.enabled or self._syncing:
            return
        slice_indices = event.dims_state.selection.slice_indices
        updates = {a: slice_indices[a] for a in self._extra_axes if a in slice_indices}
        if not updates:
            return
        self._syncing = True
        try:
            for scene_id, scene in self._scenes_by_id.items():
                if scene_id == event.scene_id:
                    continue
                current = dict(scene.dims.selection.slice_indices)
                changed = False
                for axis, value in updates.items():
                    if current.get(axis) != value:
                        current[axis] = value
                        changed = True
                if changed:
                    self._controller.update_slice_indices(scene_id, current)
        finally:
            self._syncing = False


def _resolve_spatial_axes(
    axis_labels: tuple[str, ...],
    spatial_axes: tuple[str, ...] | tuple[int, ...] | None,
) -> tuple[int, int, int]:
    """Resolve the three spatial axis indices in ``(z, y, x)`` order.

    Defaults to the last three axes when *spatial_axes* is ``None``.  Otherwise
    accepts a length-3 tuple of axis names or indices.
    """
    ndim = len(axis_labels)
    if ndim < 3:
        raise ValueError(
            f"OrthoViewer requires at least 3 axes, got {ndim}: {axis_labels!r}"
        )

    if spatial_axes is None:
        return (ndim - 3, ndim - 2, ndim - 1)

    if len(spatial_axes) != 3:
        raise ValueError(
            f"spatial_axes must have exactly 3 entries, got {len(spatial_axes)}: "
            f"{spatial_axes!r}"
        )

    label_to_index = {label: i for i, label in enumerate(axis_labels)}
    resolved: list[int] = []
    for entry in spatial_axes:
        if isinstance(entry, str):
            if entry not in label_to_index:
                raise ValueError(
                    f"Unknown spatial axis name {entry!r}. "
                    f"Available: {list(axis_labels)}"
                )
            resolved.append(label_to_index[entry])
        else:
            index = int(entry)
            if not 0 <= index < ndim:
                raise ValueError(
                    f"spatial axis index {index} out of range for {ndim} axes."
                )
            resolved.append(index)

    if len(set(resolved)) != 3:
        raise ValueError(f"spatial_axes must be distinct, got {spatial_axes!r}")
    return (resolved[0], resolved[1], resolved[2])


class OrthoViewer:
    """Four-panel orthoviewer wrapping a single CellierController.

    Creates a controller and four pre-wired scenes that share one world
    coordinate system: three orthogonal 2D slice panels (``xy``, ``xz``,
    ``yz``) and one 3D volume panel (``vol``).  No Qt objects are constructed
    here; build canvases with
    :func:`cellier.convenience.gui.build_ortho_grid_widget` when ready.

    The three spatial axes (default: the last three axis labels, treated as
    ``z, y, x``) define the slice planes.  Any remaining axes are "extra" axes
    (e.g. channel or time): they appear as sliders on every panel and, by
    default, stay synchronized across panels.

    Parameters
    ----------
    axis_labels : tuple[str, ...]
        World-axis names in order, e.g. ``("c", "z", "y", "x")``.  The number
        of labels sets the dimensionality.  Must contain at least 3 axes.
    spatial_axes : tuple[str, ...], tuple[int, ...], or None
        The three axes (names or indices, in ``z, y, x`` order) that form the
        orthogonal planes.  Defaults to the last three axes when ``None``.
    link_extra_axes : bool
        When ``True`` (default), extra (non-spatial) axis slider positions are
        kept synchronized across all four panels.
    render_config : RenderManagerConfig or None
        Render pipeline configuration passed through to the controller.
    """

    def __init__(
        self,
        axis_labels: tuple[str, ...],
        *,
        spatial_axes: tuple[str, ...] | tuple[int, ...] | None = None,
        link_extra_axes: bool = True,
        render_config: RenderManagerConfig | None = None,
    ) -> None:
        self._controller = CellierController(render_config=render_config)
        self._spatial_axes = _resolve_spatial_axes(axis_labels, spatial_axes)
        self._ndim = len(axis_labels)
        self._extra_axes = {i for i in range(self._ndim) if i not in self._spatial_axes}
        coordinate_system = CoordinateSystem(name="world", axis_labels=axis_labels)
        self._scenes = self._build_scenes(coordinate_system)
        self._syncer: _ExtraAxisSyncer | None = None
        if link_extra_axes and self._extra_axes:
            self._wire_extra_axis_sync()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scenes(self, coordinate_system: CoordinateSystem) -> dict[str, Scene]:
        s0, s1, s2 = self._spatial_axes
        # displayed axes per panel; the remaining spatial axis is sliced.
        displayed_by_key: dict[str, tuple[int, ...]] = {
            "xy": (s1, s2),
            "xz": (s0, s2),
            "yz": (s0, s1),
            "vol": (s0, s1, s2),
        }
        scenes: dict[str, Scene] = {}
        for key in _PANEL_KEYS:
            displayed = displayed_by_key[key]
            render_modes = {"3d"} if key == "vol" else {"2d"}
            slice_indices = {i: 0 for i in range(self._ndim) if i not in displayed}
            scene = Scene(
                name=key,
                dims=DimsManager(
                    coordinate_system=coordinate_system,
                    selection=AxisAlignedSelection(
                        displayed_axes=displayed,
                        slice_indices=slice_indices,
                    ),
                ),
                render_modes=render_modes,
                lighting="none",
            )
            scenes[key] = self._controller.add_scene_model(scene)
        return scenes

    def _wire_extra_axis_sync(self) -> None:
        """Subscribe the cross-panel extra-axis syncer to all panels."""
        scenes_by_id = {scene.id: scene for scene in self._scenes.values()}
        syncer = _ExtraAxisSyncer(self._controller, scenes_by_id, self._extra_axes)
        for scene in self._scenes.values():
            self._controller.on_dims_changed(
                scene.id, syncer.handle, owner_id=syncer.owner_id
            )
        self._syncer = syncer

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def controller(self) -> CellierController:
        """The underlying CellierController."""
        return self._controller

    @property
    def scenes(self) -> dict[str, Scene]:
        """The four panel scenes keyed ``"xy"``, ``"xz"``, ``"yz"``, ``"vol"``."""
        return self._scenes

    @property
    def spatial_axes(self) -> tuple[int, int, int]:
        """The three spatial axis indices in ``(z, y, x)`` order."""
        return self._spatial_axes

    @property
    def extra_axes(self) -> set[int]:
        """Indices of the non-spatial (extra) axes."""
        return set(self._extra_axes)

    @property
    def extra_axis_sync_enabled(self) -> bool:
        """Whether extra-axis positions are synced across panels."""
        return self._syncer is not None and self._syncer.enabled

    @extra_axis_sync_enabled.setter
    def extra_axis_sync_enabled(self, value: bool) -> None:
        if self._syncer is None:
            if value and self._extra_axes:
                self._wire_extra_axis_sync()
            return
        self._syncer.enabled = value

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_file(self, path: str | Path) -> None:
        """Serialize the orthoviewer model state to a JSON file.

        Captures the four scenes (dims, slice positions), visuals, data stores,
        canvas camera state, and the render pipeline configuration.  The live
        extra-axis sync wiring is *not* serialized; :meth:`from_file`
        re-establishes it.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        self._controller.to_file(path)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        link_extra_axes: bool = True,
        render_config: RenderManagerConfig | None = None,
    ) -> OrthoViewer:
        """Restore an ``OrthoViewer`` from a previously serialized file.

        The four panels are rebound by scene name (``xy``, ``xz``, ``yz``,
        ``vol``) and the spatial axes are recovered from the ``vol`` panel's
        displayed axes -- no extra metadata is stored.  Extra-axis sync is
        re-established when *link_extra_axes* is ``True``.

        Parameters
        ----------
        path : str or Path
            Path to a JSON file written by :meth:`to_file`.
        link_extra_axes : bool
            Re-subscribe the cross-panel extra-axis syncer.  Default ``True``.
        render_config : RenderManagerConfig or None
            Override the serialized render pipeline configuration.

        Returns
        -------
        OrthoViewer

        Raises
        ------
        ValueError
            If the file does not contain exactly the four expected panels.
        """
        controller = CellierController.from_file(path, render_config=render_config)
        scenes_by_name = {
            scene.name: scene for scene in controller._model.scenes.values()
        }
        expected = set(_PANEL_KEYS)
        if set(scenes_by_name) != expected or len(controller._model.scenes) != len(
            _PANEL_KEYS
        ):
            raise ValueError(
                "Expected exactly the four orthoviewer panels "
                f"{sorted(expected)} in the file, found "
                f"{sorted(scenes_by_name)}. Use CellierController.from_file "
                "directly for other models."
            )
        scenes = {key: scenes_by_name[key] for key in _PANEL_KEYS}
        vol_displayed = tuple(scenes["vol"].dims.selection.displayed_axes)
        ndim = len(scenes["vol"].dims.coordinate_system.axis_labels)

        obj = object.__new__(cls)
        obj._controller = controller
        obj._scenes = scenes
        obj._spatial_axes = vol_displayed  # type: ignore[assignment]
        obj._ndim = ndim
        obj._extra_axes = {i for i in range(ndim) if i not in vol_displayed}
        obj._syncer = None
        if link_extra_axes and obj._extra_axes:
            obj._wire_extra_axis_sync()
        return obj

    # ------------------------------------------------------------------
    # Dims control
    # ------------------------------------------------------------------

    def center_slices(self) -> None:
        """Move each panel's sliced spatial axis to the middle of the data.

        Reads the world-space extent of the loaded visuals and sets each 2D
        panel's sliced spatial axis to its midpoint.  Extra-axis positions are
        left unchanged.  Call after adding data.

        Raises
        ------
        ValueError
            If no visuals with known shapes have been added yet.
        """
        from cellier.convenience._geometry import axis_ranges_from_ortho

        ranges = axis_ranges_from_ortho(self)
        for scene in self._scenes.values():
            new_slices = dict(scene.dims.selection.slice_indices)
            updated = False
            for axis in self._spatial_axes:
                if axis in new_slices and axis in ranges:
                    low, high = ranges[axis]
                    # slice_indices are integer world coordinates.
                    new_slices[axis] = round((low + high) / 2.0)
                    updated = True
            if updated:
                self._controller.update_slice_indices(scene.id, new_slices)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_data_store(self, data: _T | UUID) -> _T:
        if isinstance(data, UUID):
            return self._controller.get_data_store(data)  # type: ignore[return-value]
        return data

    def _fan_out(self, add_one) -> dict[str, object]:
        """Call *add_one(key, scene)* for every panel and collect the results."""
        return {key: add_one(key, scene) for key, scene in self._scenes.items()}

    # ------------------------------------------------------------------
    # Visual add methods (one data store, one visual per panel)
    # ------------------------------------------------------------------

    def add_image(
        self,
        data: ImageMemoryStore | UUID,
        appearance: BaseImageAppearance | InMemoryImageAppearanceKwargs,
        name: str = "image",
    ) -> dict[str, ImageVisual]:
        """Add an in-memory image to every panel from a single data store.

        Parameters
        ----------
        data : ImageMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : BaseImageAppearance or dict
            Appearance parameters.  Accepts an ``InMemoryImageAppearance``
            instance or a plain dict with the same keys (see
            ``InMemoryImageAppearanceKwargs``).
        name : str
            Base label; each panel's visual is named ``f"{name}_{key}"``.

        Returns
        -------
        dict[str, ImageVisual]
            The per-panel visuals keyed ``"xy"``, ``"xz"``, ``"yz"``, ``"vol"``.
        """
        from cellier.visuals._image_memory import InMemoryImageAppearance

        store = self._resolve_data_store(data)
        resolved = (
            InMemoryImageAppearance.model_validate(appearance)
            if isinstance(appearance, dict)
            else appearance
        )
        return self._fan_out(
            lambda key, scene: self._controller.add_image(
                store, scene.id, resolved, f"{name}_{key}"
            )
        )

    def add_labels(
        self,
        data: LabelMemoryStore | UUID,
        appearance: BaseLabelsAppearance | InMemoryLabelsAppearanceKwargs | None = None,
        name: str = "labels",
        transform: AffineTransform | None = None,
    ) -> dict[str, LabelMemoryVisual]:
        """Add an in-memory label image to every panel from one data store.

        Parameters
        ----------
        data : LabelMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : BaseLabelsAppearance, dict, or None
            Appearance parameters.  Accepts an ``InMemoryLabelsAppearance``
            instance or a plain dict (see ``InMemoryLabelsAppearanceKwargs``).
            Defaults to ``InMemoryLabelsAppearance()`` when ``None``.
        name : str
            Base label; each panel's visual is named ``f"{name}_{key}"``.
        transform : AffineTransform or None
            Data-to-world transform.  Defaults to identity when ``None``.

        Returns
        -------
        dict[str, LabelMemoryVisual]
        """
        from cellier.visuals._label_memory import InMemoryLabelsAppearance

        store = self._resolve_data_store(data)
        resolved = (
            InMemoryLabelsAppearance.model_validate(appearance)
            if isinstance(appearance, dict)
            else appearance
        )
        return self._fan_out(
            lambda key, scene: self._controller.add_labels(
                store, scene.id, resolved, f"{name}_{key}", transform
            )
        )

    def add_mesh(
        self,
        data: MeshMemoryStore | UUID,
        appearance: MeshAppearance
        | MeshFlatAppearanceKwargs
        | MeshPhongAppearanceKwargs,
        name: str = "mesh",
        transform: AffineTransform | None = None,
    ) -> dict[str, MeshVisual]:
        """Add a mesh to every panel from a single data store.

        Parameters
        ----------
        data : MeshMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : MeshFlatAppearance, MeshPhongAppearance, or dict
            Appearance parameters.  When passing a dict, include the
            ``appearance_type`` key (``"flat"`` or ``"phong"``).
        name : str
            Base label; each panel's visual is named ``f"{name}_{key}"``.
        transform : AffineTransform or None
            Data-to-world transform.  Defaults to identity when ``None``.

        Returns
        -------
        dict[str, MeshVisual]
        """
        from pydantic import TypeAdapter

        from cellier.visuals._mesh_memory import MeshAppearance as _MeshAppearance

        store = self._resolve_data_store(data)
        resolved = (
            TypeAdapter(_MeshAppearance).validate_python(appearance)
            if isinstance(appearance, dict)
            else appearance
        )
        return self._fan_out(
            lambda key, scene: self._controller.add_mesh(
                store, scene.id, resolved, f"{name}_{key}", transform
            )
        )

    def add_points(
        self,
        data: PointsMemoryStore | UUID,
        appearance: PointsMarkerAppearance | PointsMarkerAppearanceKwargs | None = None,
        name: str = "points",
        transform: AffineTransform | None = None,
    ) -> dict[str, PointsVisual]:
        """Add a points visual to every panel from a single data store.

        Parameters
        ----------
        data : PointsMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : PointsMarkerAppearance, dict, or None
            Appearance parameters (see ``PointsMarkerAppearanceKwargs``).
            Defaults to ``PointsMarkerAppearance()`` when ``None``.
        name : str
            Base label; each panel's visual is named ``f"{name}_{key}"``.
        transform : AffineTransform or None
            Data-to-world transform.  Defaults to identity when ``None``.

        Returns
        -------
        dict[str, PointsVisual]
        """
        from cellier.visuals._points_memory import (
            PointsMarkerAppearance as _PointsMarkerAppearance,
        )

        store = self._resolve_data_store(data)
        resolved = (
            _PointsMarkerAppearance.model_validate(appearance)
            if isinstance(appearance, dict)
            else appearance
        )
        return self._fan_out(
            lambda key, scene: self._controller.add_points(
                store, scene.id, resolved, f"{name}_{key}", transform
            )
        )

    def add_lines(
        self,
        data: LinesMemoryStore | UUID,
        appearance: LinesMemoryAppearance | LinesMemoryAppearanceKwargs | None = None,
        name: str = "lines",
        transform: AffineTransform | None = None,
    ) -> dict[str, LinesVisual]:
        """Add a lines visual to every panel from a single data store.

        Parameters
        ----------
        data : LinesMemoryStore or UUID
            Backing data store or the UUID of an already-registered store.
        appearance : LinesMemoryAppearance, dict, or None
            Appearance parameters (see ``LinesMemoryAppearanceKwargs``).
            Defaults to ``LinesMemoryAppearance()`` when ``None``.
        name : str
            Base label; each panel's visual is named ``f"{name}_{key}"``.
        transform : AffineTransform or None
            Data-to-world transform.  Defaults to identity when ``None``.

        Returns
        -------
        dict[str, LinesVisual]
        """
        from cellier.visuals._lines_memory import (
            LinesMemoryAppearance as _LinesMemoryAppearance,
        )

        store = self._resolve_data_store(data)
        resolved = (
            _LinesMemoryAppearance.model_validate(appearance)
            if isinstance(appearance, dict)
            else appearance
        )
        return self._fan_out(
            lambda key, scene: self._controller.add_lines(
                store, scene.id, resolved, f"{name}_{key}", transform
            )
        )

    def add_image_multiscale(
        self,
        data: BaseDataStore | UUID,
        appearance: MultiscaleImageAppearance | MultiscaleImageAppearanceKwargs,
        name: str = "image",
        render_config: MultiscaleImageRenderConfig
        | MultiscaleImageRenderConfigKwargs
        | None = None,
        transform: AffineTransform | None = None,
    ) -> dict[str, MultiscaleImageVisual]:
        """Add a multiscale image to every panel from a single data store.

        Parameters
        ----------
        data : BaseDataStore or UUID
            Backing multiscale store or UUID of an already-registered store.
        appearance : MultiscaleImageAppearance or dict
            Appearance parameters (see ``MultiscaleImageAppearanceKwargs``).
        name : str
            Base label; each panel's visual is named ``f"{name}_{key}"``.
        render_config : MultiscaleImageRenderConfig, dict, or None
            LOD and rendering configuration (see
            ``MultiscaleImageRenderConfigKwargs``).  Uses defaults when ``None``.
        transform : AffineTransform or None
            Data-to-world transform.  Defaults to identity when ``None``.

        Returns
        -------
        dict[str, MultiscaleImageVisual]
        """
        from cellier.visuals._image import (
            MultiscaleImageAppearance as _MultiscaleImageAppearance,
        )
        from cellier.visuals._image import (
            MultiscaleImageRenderConfig as _MultiscaleImageRenderConfig,
        )

        store = self._resolve_data_store(data)
        resolved_appearance = (
            _MultiscaleImageAppearance.model_validate(appearance)
            if isinstance(appearance, dict)
            else appearance
        )
        resolved_render_config = (
            _MultiscaleImageRenderConfig.model_validate(render_config)
            if isinstance(render_config, dict)
            else render_config
        )
        return self._fan_out(
            lambda key, scene: self._controller.add_image_multiscale(
                store,
                scene.id,
                resolved_appearance,
                f"{name}_{key}",
                resolved_render_config,
                transform,
            )
        )

    def add_labels_multiscale(
        self,
        data: BaseDataStore | UUID,
        appearance: MultiscaleLabelsAppearance | MultiscaleLabelsAppearanceKwargs,
        name: str = "labels",
        render_config: MultiscaleLabelRenderConfig
        | MultiscaleLabelRenderConfigKwargs
        | None = None,
        transform: AffineTransform | None = None,
    ) -> dict[str, MultiscaleLabelVisual]:
        """Add a multiscale label image to every panel from one data store.

        Parameters
        ----------
        data : BaseDataStore or UUID
            Backing multiscale label store or UUID of an already-registered store.
        appearance : MultiscaleLabelsAppearance or dict
            Appearance parameters (see ``MultiscaleLabelsAppearanceKwargs``).
        name : str
            Base label; each panel's visual is named ``f"{name}_{key}"``.
        render_config : MultiscaleLabelRenderConfig, dict, or None
            LOD and rendering configuration (see
            ``MultiscaleLabelRenderConfigKwargs``).  Uses defaults when ``None``.
        transform : AffineTransform or None
            Data-to-world transform.  Defaults to identity when ``None``.

        Returns
        -------
        dict[str, MultiscaleLabelVisual]
        """
        from cellier.visuals._labels import (
            MultiscaleLabelRenderConfig as _MultiscaleLabelRenderConfig,
        )
        from cellier.visuals._labels import (
            MultiscaleLabelsAppearance as _MultiscaleLabelsAppearance,
        )

        store = self._resolve_data_store(data)
        resolved_appearance = (
            _MultiscaleLabelsAppearance.model_validate(appearance)
            if isinstance(appearance, dict)
            else appearance
        )
        resolved_render_config = (
            _MultiscaleLabelRenderConfig.model_validate(render_config)
            if isinstance(render_config, dict)
            else render_config
        )
        return self._fan_out(
            lambda key, scene: self._controller.add_labels_multiscale(
                store,
                scene.id,
                resolved_appearance,
                f"{name}_{key}",
                resolved_render_config,
                transform,
            )
        )

    def add_multichannel_image(
        self,
        data: ImageMemoryStore | UUID,
        channel_axis: int,
        channels: dict[int, ChannelAppearance | ChannelAppearanceKwargs],
        name: str = "multichannel_image",
        max_channels_2d: int = 8,
        max_channels_3d: int = 4,
    ) -> dict[str, MultichannelImageVisual]:
        """Add an in-memory multichannel image to every panel from one store.

        Parameters
        ----------
        data : ImageMemoryStore or UUID
            Backing data store or UUID of an already-registered store.
        channel_axis : int
            Data axis index for the channel dimension.
        channels : dict[int, ChannelAppearance or dict]
            Per-channel appearance keyed by channel index (see
            ``ChannelAppearanceKwargs``).
        name : str
            Base label; each panel's visual is named ``f"{name}_{key}"``.
        max_channels_2d : int
            Maximum simultaneous 2D channel nodes.
        max_channels_3d : int
            Maximum simultaneous 3D channel nodes.

        Returns
        -------
        dict[str, MultichannelImageVisual]
        """
        from cellier.visuals._channel_appearance import (
            ChannelAppearance as _ChannelAppearance,
        )

        store = self._resolve_data_store(data)
        resolved_channels = {
            k: (_ChannelAppearance.model_validate(v) if isinstance(v, dict) else v)
            for k, v in channels.items()
        }
        return self._fan_out(
            lambda key, scene: self._controller.add_multichannel_image(
                store,
                scene.id,
                channel_axis,
                resolved_channels,
                f"{name}_{key}",
                max_channels_2d,
                max_channels_3d,
            )
        )

    def add_multichannel_image_multiscale(
        self,
        data: BaseDataStore | UUID,
        channel_axis: int,
        channels: dict[int, ChannelAppearance | ChannelAppearanceKwargs],
        name: str = "multichannel_image",
        render_config: MultiscaleImageRenderConfig
        | MultiscaleImageRenderConfigKwargs
        | None = None,
        transform: AffineTransform | None = None,
        max_channels_2d: int = 8,
        max_channels_3d: int = 4,
    ) -> dict[str, MultichannelMultiscaleImageVisual]:
        """Add a multiscale multichannel image to every panel from one store.

        Parameters
        ----------
        data : BaseDataStore or UUID
            Backing multiscale store or UUID of an already-registered store.
        channel_axis : int
            Data axis index for the channel dimension.
        channels : dict[int, ChannelAppearance or dict]
            Per-channel appearance keyed by channel index (see
            ``ChannelAppearanceKwargs``).
        name : str
            Base label; each panel's visual is named ``f"{name}_{key}"``.
        render_config : MultiscaleImageRenderConfig, dict, or None
            LOD and rendering configuration (see
            ``MultiscaleImageRenderConfigKwargs``).  Uses defaults when ``None``.
        transform : AffineTransform or None
            Data-to-world transform.  Defaults to identity when ``None``.
        max_channels_2d : int
            Maximum simultaneous 2D channel nodes.
        max_channels_3d : int
            Maximum simultaneous 3D channel nodes.

        Returns
        -------
        dict[str, MultichannelMultiscaleImageVisual]
        """
        from cellier.visuals._channel_appearance import (
            ChannelAppearance as _ChannelAppearance,
        )
        from cellier.visuals._image import (
            MultiscaleImageRenderConfig as _MultiscaleImageRenderConfig,
        )

        store = self._resolve_data_store(data)
        resolved_channels = {
            k: (_ChannelAppearance.model_validate(v) if isinstance(v, dict) else v)
            for k, v in channels.items()
        }
        resolved_render_config = (
            _MultiscaleImageRenderConfig.model_validate(render_config)
            if isinstance(render_config, dict)
            else render_config
        )
        return self._fan_out(
            lambda key, scene: self._controller.add_multichannel_image_multiscale(
                store,
                scene.id,
                channel_axis,
                resolved_channels,
                f"{name}_{key}",
                resolved_render_config,
                transform,
                max_channels_2d,
                max_channels_3d,
            )
        )
