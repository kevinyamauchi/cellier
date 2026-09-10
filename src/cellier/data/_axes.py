"""Building a datastore's coordinate systems from what the store can say.

``Axis.axis_type`` has no default on purpose: a channel or time axis typed
``"space"`` is not caught where it is written, but much later, by
``from_axis_map`` refusing to map axes whose types disagree.  So a store never
guesses a type out of nothing.  There are exactly three ways it gets one, in
descending order of authority:

1. **The dataset says.**  The OME-Zarr readers pass ``axis_types`` straight
   from the NGFF metadata, and an empty ``type`` there raises rather than
   defaulting -- the metadata had a slot for it and left it blank, which is a
   defect in the dataset.
2. **The caller says.**  ``axis_names=`` plus ``axis_types=`` on an in-memory
   store.  With ``axis_names`` alone, :func:`axis_types_from_names` applies one
   small documented rule.
3. **The scene says.**  A store added to a scene with no systems of its own
   takes the world's trailing axes, name, type and unit alike.  That is not a
   guess: the world was declared explicitly by the caller, and inheriting from
   it is what makes the ``data -> world`` transform typecheck by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cellier.transform_v2 import (
    AffineTransform,
    Axis,
    AxisType,
    DataCoordinateSystem,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

    from cellier.transform_v2 import CoordinateSystem, WorldCoordinateSystem


#: Axis names that carry a non-spatial type when only names are supplied.
_TYPE_BY_NAME: dict[str, AxisType] = {
    "t": "time",
    "time": "time",
    "c": "channel",
    "channel": "channel",
}


def axis_types_from_names(names: Sequence[str]) -> tuple[AxisType, ...]:
    """Apply the name-based type rule to a sequence of axis names.

    ``t`` / ``time`` are time, ``c`` / ``channel`` are channel, everything
    else is space.  Case-insensitive.  This is used **only** when a caller
    supplies ``axis_names`` without ``axis_types``; it is a convenience for
    the conventional names, not a fallback for the general case.

    Parameters
    ----------
    names : Sequence[str]
        The axis names, in data order.

    Returns
    -------
    tuple[AxisType, ...]
        One type per name.
    """
    return tuple(_TYPE_BY_NAME.get(name.lower(), "space") for name in names)


def build_axes(
    names: Sequence[str],
    types: Sequence[AxisType] | None = None,
    units: Sequence[str | None] | None = None,
) -> tuple[Axis, ...]:
    """Build axes from names, with types and units where they are known.

    Parameters
    ----------
    names : Sequence[str]
        Axis names, in data order.
    types : Sequence[AxisType] or None
        Axis types, one per name.  ``None`` applies
        :func:`axis_types_from_names`.
    units : Sequence[str | None] or None
        Physical units, one per name.  ``None`` leaves every unit unset.

    Returns
    -------
    tuple[Axis, ...]
        The axes, each with a fresh id.

    Raises
    ------
    ValueError
        If *types* or *units* has a different length from *names*, or if a
        type is empty.  An empty type is a dataset defect, not something to
        default: it is named here with the axis it came from.
    """
    resolved_types = axis_types_from_names(names) if types is None else tuple(types)
    if len(resolved_types) != len(names):
        raise ValueError(
            f"axis_types must have one entry per axis name; got "
            f"{len(resolved_types)} types for {len(names)} names {list(names)}."
        )
    resolved_units: tuple[str | None, ...] = (
        (None,) * len(names) if units is None else tuple(units)
    )
    if len(resolved_units) != len(names):
        raise ValueError(
            f"axis_units must have one entry per axis name; got "
            f"{len(resolved_units)} units for {len(names)} names {list(names)}."
        )
    axes: list[Axis] = []
    for name, axis_type, unit in zip(names, resolved_types, resolved_units):
        if not axis_type:
            raise ValueError(
                f"Axis '{name}' has an empty axis_type.  OME-NGFF has a slot "
                f"for it and this dataset left it blank; there is no honest "
                f"default among {AxisType.__args__}.  Fix the metadata, or "
                f"pass axis_types= explicitly when constructing the store."
            )
        axes.append(Axis(name=name, axis_type=axis_type, unit=unit or None))
    return tuple(axes)


def data_coordinate_system(
    datastore_id: UUID,
    axes: Sequence[Axis],
    name: str,
) -> DataCoordinateSystem:
    """Wrap axes into a datastore's coordinate system.

    Parameters
    ----------
    datastore_id : UUID
        The owning store's id.
    axes : Sequence[Axis]
        The axes, in data order.
    name : str
        Human-readable name for the system.

    Returns
    -------
    DataCoordinateSystem
        The system.
    """
    return DataCoordinateSystem(name=name, axes=tuple(axes), datastore_id=datastore_id)


def level_coordinate_systems(
    datastore_id: UUID,
    axes: Sequence[Axis],
    n_levels: int,
    name: str,
) -> list[DataCoordinateSystem]:
    """Build one coordinate system per resolution level.

    Every level shares the axis *names*, *types* and *units* -- they describe
    the same physical quantities -- but each is a distinct system with fresh
    axis ids, because a level-2 voxel is not a level-0 voxel and a transform
    between them is exactly what says so.

    Parameters
    ----------
    datastore_id : UUID
        The owning store's id.
    axes : Sequence[Axis]
        The level-0 axes.  Levels 1 upward copy their descriptions.
    n_levels : int
        How many levels the pyramid has.
    name : str
        Base name; each level gets ``f"{name}_level{k}"``.

    Returns
    -------
    list[DataCoordinateSystem]
        One system per level, finest first.
    """
    systems = [data_coordinate_system(datastore_id, axes, f"{name}_level0")]
    for level in range(1, n_levels):
        systems.append(
            data_coordinate_system(
                datastore_id,
                tuple(
                    Axis(name=axis.name, axis_type=axis.axis_type, unit=axis.unit)
                    for axis in axes
                ),
                f"{name}_level{level}",
            )
        )
    return systems


def identity_transform(
    input_coordinate_system: CoordinateSystem,
    output_coordinate_system: CoordinateSystem,
    name: str | None = None,
) -> AffineTransform:
    """Build the positional identity between two equal-rank systems.

    Axis *i* maps to axis *i*, scale 1, translation 0.  The correspondence is
    still stated rather than assumed -- ``from_axis_map`` requires it -- and
    the two systems' axis types must agree, which is the check that catches a
    world and a dataset that disagree about what their axes are.

    Parameters
    ----------
    input_coordinate_system : CoordinateSystem
        The system to map from.
    output_coordinate_system : CoordinateSystem
        The system to map to.  Must have the same rank.
    name : str or None
        Optional name for the transform.

    Returns
    -------
    AffineTransform
        The identity between the two systems.

    Raises
    ------
    ValueError
        If the ranks differ.
    """
    if input_coordinate_system.ndim != output_coordinate_system.ndim:
        raise ValueError(
            f"An identity transform needs equal ranks; "
            f"'{input_coordinate_system.name}' has "
            f"{input_coordinate_system.ndim} axes and "
            f"'{output_coordinate_system.name}' has "
            f"{output_coordinate_system.ndim}."
        )
    return AffineTransform.from_axis_map(
        input_coordinate_system,
        output_coordinate_system,
        axis_map={
            axis.id: output_coordinate_system.axes[index].id
            for index, axis in enumerate(input_coordinate_system.axes)
        },
        name=name,
    )


def data_axes_from_world(
    world_coordinate_system: WorldCoordinateSystem,
    ndim: int,
    declared: Mapping[int, tuple[str, AxisType]] | None = None,
) -> tuple[Axis, ...]:
    """Take a store's axes from the trailing axes of the world.

    The fallback of last resort, for a store that carries no axis metadata --
    a bare ``ImageMemoryStore(data=arr)``.  Trailing rather than leading
    because that is the alignment the rest of cellier already assumes: a
    scene's initial ``displayed_axes`` is the last two or three world axes,
    and a dataset that is a sub-rank of the world is a spatial volume missing
    the leading batch-like axes, not the reverse.

    Names, types and units are inherited from the world axes, so the
    ``data -> world`` transform this store gets typechecks by construction.
    The ids are fresh: these are the axes of a different system.

    *declared* covers the axes a store has that the world does not -- a
    multichannel visual's channel axis is the shipping case, and the visual
    knows which one it is.  Those are taken from *declared* and the rest fall
    into the trailing world window around them.

    Parameters
    ----------
    world_coordinate_system : WorldCoordinateSystem
        The scene's world.
    ndim : int
        The store's rank.
    declared : Mapping[int, tuple[str, AxisType]] or None
        ``{data axis index: (name, axis_type)}`` for axes that have no world
        counterpart.

    Returns
    -------
    tuple[Axis, ...]
        ``ndim`` axes.

    Raises
    ------
    ValueError
        If, after removing the declared axes, the store still has more axes
        than the world.
    """
    declared = dict(declared or {})
    inherited = ndim - len(declared)
    if inherited > world_coordinate_system.ndim:
        raise ValueError(
            f"A data store with {ndim} axes ({len(declared)} of them declared) "
            f"cannot take the rest from a world with only "
            f"{world_coordinate_system.ndim} axes "
            f"({world_coordinate_system.axis_names()}).  Give the scene more "
            f"axes, or construct the store with explicit axis_names."
        )
    window = list(
        world_coordinate_system.axes[world_coordinate_system.ndim - inherited :]
    )
    axes: list[Axis] = []
    for index in range(ndim):
        if index in declared:
            name, axis_type = declared[index]
            axes.append(Axis(name=name, axis_type=axis_type))
        else:
            source = window.pop(0)
            axes.append(
                Axis(name=source.name, axis_type=source.axis_type, unit=source.unit)
            )
    return tuple(axes)


def install_level_systems(
    store: object,
    names: Sequence[str],
    types: Sequence[AxisType] | None = None,
    units: Sequence[str | None] | None = None,
) -> None:
    """Give a multi-level store one coordinate system per resolution level.

    A no-op when the store already has systems, so it is safe to call from
    ``model_post_init``: a store restored from JSON keeps the ids its stored
    transforms name, and only a freshly read one mints new ones.

    ``level_transforms`` is deliberately untouched.  The OME-Zarr readers
    derive theirs from NGFF metadata and still hold them as v1 transforms;
    converting them is the ``data -> world`` phase's job, not this one's.

    Parameters
    ----------
    store : object
        The datastore.  Must expose ``id``, ``name`` and ``n_levels``.
    names : Sequence[str]
        Axis names in data order.
    types : Sequence[AxisType] or None
        Axis types.  ``None`` applies :func:`axis_types_from_names`.
    units : Sequence[str | None] or None
        Physical units.
    """
    if store.data_coordinate_systems:
        return
    axes = build_axes(names, types, units)
    store.data_coordinate_systems = level_coordinate_systems(
        store.id, axes, int(getattr(store, "n_levels", 1)), store.name
    )


def default_data_to_world(
    data_coordinate_system: DataCoordinateSystem,
    world_coordinate_system: WorldCoordinateSystem,
) -> AffineTransform:
    """Build the ``data -> world`` transform a visual gets when none is given.

    D18 forbids a coordinate-system-less identity, so there is no default on
    the model field: the controller builds this one at ``add_visual``, when
    both systems are finally known.

    The correspondence is by **trailing position**, the same alignment the
    rest of cellier already assumes -- a scene's initial ``displayed_axes`` is
    the last two or three world axes.  The two ranks are usually equal, in
    which case this is the identity on indices, and the two ways they can
    differ have one honest reading each:

    * *fewer data axes than world axes.*  The leading world axes are
      **broadcast**: a 3-D volume in a ``TZYX`` world exists at every ``T``,
      which is not the same claim as sitting at ``T = 0`` (D25).
    * *more data axes than world axes.*  The leading data axes are projected
      away -- a multichannel store's channel axis is composited by the visual
      and never reaches the world.  ``from_axis_map`` cannot express a dropped
      input axis, so the matrix is built directly.

    Parameters
    ----------
    data_coordinate_system : DataCoordinateSystem
        The store's level-0 system.
    world_coordinate_system : WorldCoordinateSystem
        The scene's world.

    Returns
    -------
    AffineTransform
        A scale-1, translation-0 transform between the two systems.
    """
    n_data = data_coordinate_system.ndim
    n_world = world_coordinate_system.ndim
    if n_data <= n_world:
        offset = n_world - n_data
        return AffineTransform.from_axis_map(
            data_coordinate_system,
            world_coordinate_system,
            axis_map={
                data_coordinate_system.axes[index].id: world_coordinate_system.axes[
                    index + offset
                ].id
                for index in range(n_data)
            },
            broadcast_output_axes=[
                world_coordinate_system.axes[index].id for index in range(offset)
            ],
            name="data_to_world",
        )
    dropped = n_data - n_world
    matrix = np.zeros((n_world + 1, n_data + 1))
    matrix[n_world, n_data] = 1.0
    for world_index in range(n_world):
        matrix[world_index, world_index + dropped] = 1.0
    return AffineTransform.from_matrix(
        matrix,
        data_coordinate_system,
        world_coordinate_system,
        name="data_to_world",
    )


def transform_from_v1(
    matrix: np.ndarray,
    data_coordinate_system: DataCoordinateSystem,
    world_coordinate_system: WorldCoordinateSystem,
) -> AffineTransform:
    """Name the endpoints of a bare ``data -> world`` matrix.

    A **migration affordance**, and the only place a v1 transform crosses into
    the new layer.  ``cellier.transform.AffineTransform`` has no notion of what
    it maps from or to, so a caller holding one -- a script, a geff file's
    per-axis scale, a test written before this migration -- has stated the
    numbers but not the spaces.  This attaches the two systems the controller
    already knows and changes nothing else, float32 to float64 aside.

    It goes away with v1 itself.  New code should build the transform with
    ``AffineTransform.from_axis_map``, which states the axis correspondence
    rather than assuming it is positional.

    Parameters
    ----------
    matrix : np.ndarray
        A square ``(n + 1, n + 1)`` homogeneous matrix in data-axis order.
    data_coordinate_system : DataCoordinateSystem
        The store's level-0 system.
    world_coordinate_system : WorldCoordinateSystem
        The scene's world.

    Returns
    -------
    AffineTransform
        The same map, with both endpoints named.

    Raises
    ------
    ValueError
        If the matrix's rank does not match both systems.  A v1 transform is
        square and positional, so there is nothing to align against when the
        two systems disagree about how many axes there are.
    """
    matrix = np.asarray(matrix, dtype=float)
    rank = matrix.shape[0] - 1
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"A v1 transform matrix is square; got shape {matrix.shape}.")
    if rank != data_coordinate_system.ndim or rank != world_coordinate_system.ndim:
        raise ValueError(
            f"A rank-{rank} v1 transform cannot be placed between "
            f"'{data_coordinate_system.name}' ({data_coordinate_system.ndim} "
            f"axes) and '{world_coordinate_system.name}' "
            f"({world_coordinate_system.ndim} axes): it is positional and "
            f"square, so there is no correspondence to infer.  Build the "
            f"transform with AffineTransform.from_axis_map instead."
        )
    return AffineTransform.from_matrix(
        matrix,
        data_coordinate_system,
        world_coordinate_system,
        name="data_to_world",
    )


def store_level_transforms(store: object) -> list[AffineTransform]:
    """The level ``k`` -> level ``0`` transforms of a store, as v2.

    A pyramid's level transforms are model state -- they are intrinsic to the
    data and the readers derive them from OME-NGFF metadata -- but the
    OME-Zarr and multiscale-zarr stores still hold theirs as v1 matrices,
    which the multiscale *visuals* also consume for their brick grids and
    shader uniforms.  Until that field moves, this derives the v2 form the
    region pull-back needs.

    It is not a fresh mint: the endpoints are the store's **stored**
    per-level coordinate systems, so the derived transforms are as stable as
    they are.

    Parameters
    ----------
    store : object
        A datastore.  Must expose ``data_coordinate_systems``; a
        ``level_transforms`` list, v1 or v2, is used where present.

    Returns
    -------
    list[AffineTransform]
        One transform per level, finest first, index 0 the identity.  Empty
        when the store has no coordinate systems.
    """
    systems = list(getattr(store, "data_coordinate_systems", []))
    if not systems:
        return []
    existing = list(getattr(store, "level_transforms", []))
    if existing and all(
        isinstance(transform, AffineTransform) for transform in existing
    ):
        return existing
    transforms: list[AffineTransform] = []
    for level, system in enumerate(systems):
        if level < len(existing):
            matrix = np.asarray(existing[level].matrix, dtype=float)
            transforms.append(
                AffineTransform.from_matrix(
                    matrix, system, systems[0], name=f"level{level}_to_level0"
                )
            )
        else:
            transforms.append(identity_transform(system, systems[0]))
    return transforms
