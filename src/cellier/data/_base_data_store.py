"""Base class for data stores."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Annotated, Any, ClassVar
from uuid import uuid4

import numpy as np
from psygnal import EventedModel
from pydantic import UUID4, AfterValidator, Field, model_validator

from cellier.data._axes import (
    build_axes,
    data_coordinate_system,
    identity_transform,
)
from cellier.data._dataset_info import DatasetInfo, RowSection
from cellier.transform import (  # noqa: TC001
    AffineTransform,
    DataCoordinateSystem,
)
from cellier.transform._axis import AxisSampling  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The extents of an axis that carries no data at all.
#:
#: ``axis_extents`` returns this for a geometry store holding zero
#: vertices.  It is distinct from a zero-width extent: an empty store does
#: not occupy the point ``{0}``, it occupies nothing, and a consumer taking
#: a union across visuals must skip it rather than pull the union to zero.
NO_EXTENT = None


def gridded_axis_extents(shape: Sequence[int]) -> tuple[tuple[float, float], ...]:
    """Per-axis extents of a regular grid, on the edge convention.

    A voxel is centred on its integer index and spans
    ``[i - 0.5, i + 0.5)``, so an axis of ``size`` voxels spans
    ``[-0.5, size - 0.5]`` -- half a voxel beyond the first and last
    centres.  This is the convention
    :func:`~cellier.render.visuals._slicing.round_world_to_voxel` documents
    and the whole render layer already uses; returning the centres
    ``(0, size - 1)`` instead would understate every axis by half a voxel at
    each end.

    Parameters
    ----------
    shape : Sequence[int]
        The level-0 shape, one entry per data axis.

    Returns
    -------
    tuple[tuple[float, float], ...]
        ``(low, high)`` per axis, in level-0 data coordinates.
    """
    return tuple((-0.5, float(size) - 0.5) for size in shape)


def geometry_axis_extents(
    positions: np.ndarray,
) -> tuple[tuple[float, float], ...] | None:
    """Per-axis extents of a point cloud, as its bounding box.

    Geometry has no grid to take a shape from, so its extent is simply the
    range its coordinates occupy.  Unlike the gridded case there is no
    half-voxel padding: a vertex is a point, not a cell.

    Parameters
    ----------
    positions : np.ndarray
        ``(n_vertices, ndim)`` array in level-0 data coordinates.

    Returns
    -------
    tuple[tuple[float, float], ...] or None
        ``(low, high)`` per axis, or ``NO_EXTENT`` when the store holds no
        vertices -- an empty store occupies nothing, which is not the same
        as occupying zero width at the origin.
    """
    if positions.ndim != 2:
        raise ValueError(
            f"positions must be a 2D (n_vertices, ndim) array to have "
            f"per-axis extents; got shape {positions.shape}."
        )
    if positions.shape[0] == 0:
        return NO_EXTENT
    lows = np.min(positions, axis=0)
    highs = np.max(positions, axis=0)
    return tuple(
        (float(low), float(high)) for low, high in zip(lows, highs, strict=True)
    )


class BaseDataStore(EventedModel):
    """The base class for all DataStores.

    Parameters
    ----------
    id : UUID4
        The unique identifier for the data store.
        The default value is a UUID4 generated hex string.
    name : str
        The name of the data store.
    data_coordinate_systems : list[DataCoordinateSystem]
        The store's intrinsic (voxel) coordinate systems, one per resolution
        level, index ``0`` being the finest.  A single-resolution store has
        one entry.

        **Stored, not derived.**  Transforms serialize their endpoints as
        UUIDs, so a system rebuilt on load would mint fresh ids and every
        transform naming it would point at nothing.

        The default is an **empty list**: a store with no axis metadata does
        not invent one, because ``Axis.axis_type`` has no honest default.
        Concrete stores that can say -- the OME-Zarr readers -- populate it at
        construction, and the in-memory stores take the ``axis_names`` /
        ``axis_types`` shorthand.  A store that reaches
        ``CellierController.add_visual`` still empty has one derived from the
        scene's world axes; see ``CellierController._ensure_data_coordinate_systems``.
    level_scales : list[tuple[float, ...]]
        Per-level, per-axis scale of level ``k`` voxels in level ``0`` voxels.
        ``level_scales[0]`` is all ones.  Empty for a single-level store.

        **This is where a pyramid's geometry is stated.**  The readers derive
        it from OME-NGFF metadata; it is raw numbers, which is all a store can
        know on its own.
    level_translations : list[tuple[float, ...]]
        The offset half of the same, in level-0 voxels.
        ``level_translations[0]`` is all zeros.
    level_transforms : list[AffineTransform]
        Level ``k`` voxel space -> level ``0`` voxel space, one per entry in
        ``data_coordinate_systems``.  Index ``0`` is the identity.  Empty
        alongside an empty ``data_coordinate_systems``.

        **Derived, not authored.**  Built from ``level_scales`` /
        ``level_translations`` by ``install_level_transforms`` as soon as the
        store has its coordinate systems -- from its own axis metadata, or
        from the scene's world.  It is stored rather than recomputed on read
        because a transform serializes its endpoints as ids, and a rebuilt
        one would name systems that no longer exist.

    Attributes
    ----------
    id : str
        The unique identifier for the data store.
    """

    # store a UUID to identify this specific scene.
    id: UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))] = (
        Field(frozen=True, default_factory=lambda: uuid4())
    )
    name: str = "data store"
    data_coordinate_systems: list[DataCoordinateSystem] = Field(default_factory=list)
    level_scales: list[tuple[float, ...]] = Field(default_factory=list)
    level_translations: list[tuple[float, ...]] = Field(default_factory=list)
    level_transforms: list[AffineTransform] = Field(default_factory=list)

    AXIS_SAMPLING: ClassVar[AxisSampling] = "continuous"
    """Whether this kind of store's axes are sample-indexed.

    A voxel grid is sample-indexed by construction, so the gridded stores
    override this to ``"discrete"`` and their callers never have to say so.
    Geometry stores keep ``"continuous"``: a vertex coordinate is a measured
    position in general, and a store whose column happens to hold acquisition
    frame numbers says so per axis when it is constructed.

    Read only by :meth:`_expand_axis_shorthand`, i.e. by the ``axis_names=``
    shorthand.  A caller who builds ``data_coordinate_systems`` by hand sets
    ``Axis.sampling`` there instead, and this is not consulted.
    """

    @model_validator(mode="before")
    @classmethod
    def _expand_axis_shorthand(cls, data: Any) -> Any:
        """Turn the ``axis_*`` shorthand into a coordinate system.

        The ergonomic path for a single-level store: naming the axes is one
        keyword rather than a hand-built ``DataCoordinateSystem`` and a UUID.
        Multi-level stores are not covered -- their level transforms carry a
        downsampling factor that no shorthand can supply -- and neither are
        the stores that already declare ``axis_names`` as a field of their
        own (the OME-Zarr readers, which build their systems from NGFF
        metadata in ``from_path``).
        """
        if not isinstance(data, dict) or "axis_names" in cls.model_fields:
            return data
        names = data.pop("axis_names", None)
        types = data.pop("axis_types", None)
        units = data.pop("axis_units", None)
        sampling = data.pop("axis_sampling", None)
        if names is None:
            if types is not None or units is not None or sampling is not None:
                raise ValueError(
                    "axis_types / axis_units / axis_sampling need axis_names "
                    "alongside them; there is nothing to attach them to."
                )
            return data
        if data.get("data_coordinate_systems"):
            raise ValueError(
                "Pass either axis_names or data_coordinate_systems, not both."
            )
        store_id = data.get("id")
        if store_id is None:
            store_id = uuid4()
        elif isinstance(store_id, str):
            store_id = uuid.UUID(store_id, version=4)
        data["id"] = store_id
        name = data.get("name") or cls.model_fields["name"].default
        axes = build_axes(
            names,
            types,
            units,
            cls.AXIS_SAMPLING if sampling is None else sampling,
        )
        system = data_coordinate_system(store_id, axes, name)
        data["data_coordinate_systems"] = [system]
        data["level_transforms"] = [identity_transform(system, system)]
        return data

    @property
    def data_coordinate_system(self) -> DataCoordinateSystem:
        """The level-0 (intrinsic) coordinate system.

        Returns
        -------
        DataCoordinateSystem
            The finest-resolution system.

        Raises
        ------
        ValueError
            If the store has no coordinate systems.  The message names the
            two ways to supply them, because this is the error a caller who
            built a bare in-memory store outside a viewer will hit.
        """
        if not self.data_coordinate_systems:
            raise ValueError(
                f"Data store '{self.name}' has no data_coordinate_systems.  "
                f"Pass axis_names= (and axis_types= for any non-spatial axis) "
                f"when constructing it, or add it to a scene, which derives "
                f"them from the scene's world axes."
            )
        return self.data_coordinate_systems[0]

    def set_data_coordinate_systems(
        self,
        systems: list[DataCoordinateSystem],
        level_transforms: list[AffineTransform] | None = None,
    ) -> None:
        """Install the store's coordinate systems and level transforms.

        Assignment rather than construction because the systems are sometimes
        derived from the scene the store is added to, which the store cannot
        see at construction time.

        Parameters
        ----------
        systems : list[DataCoordinateSystem]
            One per resolution level, finest first.
        level_transforms : list[AffineTransform] or None
            Level ``k`` -> level ``0``.  ``None`` is allowed only for a
            single-level store, where the transform is the identity.

        Raises
        ------
        ValueError
            If *systems* is empty, if *level_transforms* has a different
            length, or if a multi-level store is given no level transforms --
            there is no downsampling factor to infer them from here.
        """
        if not systems:
            raise ValueError(
                "set_data_coordinate_systems needs at least the level-0 system."
            )
        if level_transforms is None:
            if len(systems) > 1:
                raise ValueError(
                    f"A {len(systems)}-level store needs explicit "
                    f"level_transforms; only a single-level store's transform "
                    f"is the identity."
                )
            level_transforms = [identity_transform(systems[0], systems[0])]
        if len(level_transforms) != len(systems):
            raise ValueError(
                f"level_transforms must have one entry per coordinate system; "
                f"got {len(level_transforms)} for {len(systems)} systems."
            )
        self.data_coordinate_systems = list(systems)
        self.level_transforms = list(level_transforms)

    # -- extents ---------------------------------------------------------

    @property
    def axis_extents(self) -> tuple[tuple[float, float], ...] | None:
        """Per-axis ``(low, high)`` extents in **level-0** data coordinates.

        The one number the world-extent of a visual is built from.  Every
        consumer -- the axis ranges a slider is sized from, and the
        out-of-domain check that decides a visual has no data at a given
        slice position -- reads it through the visual's own transform:

        .. code-block:: text

            world extent on an axis = transform.map(store.axis_extents)

        Stating it here, once, is what lets that rule have no special case:
        a gridded store and a point cloud answer the same question by
        different rules, and a transform maps either without knowing which
        it was given.

        **Level 0 always.**  Other resolution levels derive from it through
        ``level_transforms``, and a per-level extent would be several ways
        to say one thing.

        There are exactly two implementations.  Gridded stores use
        :func:`gridded_axis_extents` -- the **edge** convention,
        ``(-0.5, size - 0.5)``, because a voxel is centred on its index.
        Geometry stores use :func:`geometry_axis_extents` -- the bounding
        box of their vertices, with no padding, because a vertex is a point.

        **Staleness is not defended against.**  A geometry store whose
        ``positions`` are reassigned may report an extent computed from the
        previous array.  This is a knowing limitation of this pass, not an
        oversight; nothing subscribes to the field to invalidate a cache.

        Returns
        -------
        tuple[tuple[float, float], ...] or None
            One ``(low, high)`` pair per data axis, or ``NO_EXTENT`` when
            the store holds no data at all.

        Raises
        ------
        NotImplementedError
            If a concrete store has not implemented it.  There is no
            default: a store that cannot say what it spans must say so,
            rather than inventing an extent that consumers would size a
            slider from.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement axis_extents.  A "
            f"gridded store returns gridded_axis_extents(self.level_shapes[0]); "
            f"a geometry store returns geometry_axis_extents(self.positions)."
        )

    # ── Self-description ────────────────────────────────────────────────

    DATASET_INFO_LABEL: ClassVar[str] = "data store"
    """What this kind of store calls itself in the ``Store type`` row.

    A human-readable name rather than the ``store_type`` discriminator: the
    row is read by a person, and ``"in-memory points"`` says more than
    ``"points_memory"``.
    """

    def dataset_info(self) -> DatasetInfo:
        """Describe what this store holds, for display in a dataset-info widget.

        Returns the store's identity only.  Each concrete store overrides
        this to append what is specific to it -- shapes and scale levels for
        an image, node and edge counts for a graph -- so a new store type
        gains a populated widget by implementing one method, with nothing to
        register in the GUI layer.

        Every implementation is **cheap**: it reads metadata the store
        already holds and never triggers a read of the underlying array.
        Statistics that would require one (an image's value range, a label
        image's unique labels) are reported only where the data is already
        resident in memory.

        Returns
        -------
        DatasetInfo
            The sections to draw, in display order.
        """
        return DatasetInfo(sections=[RowSection(None, self._identity_rows())])

    def _identity_rows(self) -> list[tuple[str, str]]:
        """The ``Name``/``Store type`` rows every store's block opens with."""
        return [
            ("Name", self.name),
            ("Store type", self.DATASET_INFO_LABEL),
        ]
