"""The coordinate systems a render visual places its geometry with.

Today a node matrix is ``visual.transform.select_axes(displayed_axes)``
followed by an axis reversal.  ``select_axes`` extracts a square sub-block,
which is only correct when the linear block is block-diagonal with respect to
the displayed set, and on an unequal-rank transform it does not even raise --
it reads the homogeneous row and returns a plausible wrong matrix (design 3.8
finding 2).

The correct derivation composes three transforms (design 3.9):

    visual -> data   the collapsed voxel indices and the window origin
    data -> world    the visual's own transform
    world -> rendered  the inverse of the canvas embedding

and only the ``(z, y, x) -> (x, y, z)`` reversal at the end survives from the
old path.  :class:`RenderSpaces` is the bundle of systems that composition
needs, built by the controller (which owns all of them) and handed to each
render visual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from transformnd.transforms.affine import Affine

from cellier._rounding import round_half_up
from cellier.transform import AffineTransform, BaseTransform, ConvexRegion
from cellier.transform._geometry_ops import NonAffineTransformError
from cellier.transform._region import half_spaces_from_arrays

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cellier.transform import (
        DataCoordinateSystem,
        RegionSelection,
        RenderedCoordinateSystem,
        VisualCoordinateSystem,
        WorldCoordinateSystem,
    )


# ---------------------------------------------------------------------------
# Axis selection and permutation on plain value sequences
# ---------------------------------------------------------------------------
#
# These operate on shape tuples, scale vectors and translation vectors -- not
# on transforms.  They lived in v1's ``_axis_order`` module and shared
# their names with the ``AffineTransform`` methods of the same name, which was
# a coincidence: rank-reducing a transform and subsetting a shape tuple are
# different operations.  The methods went with v1; these moved here, next to
# the fetch-order invariant they serve (R8.1).
#
# ``select_axes`` picks a subset of entries from a per-data-axis sequence.
# The result preserves the order in ``axes`` and is therefore still in *data
# axis order* over the selected subset.
#
# ``swap_axes`` reorders an entire sequence by an explicit permutation.  Use
# it to convert between data-axis order and shader order.  The permutation is
# never hardcoded here -- the caller passes a literal tuple, so the assumption
# is visible at the call site.

T = TypeVar("T")


def select_axes(values: Sequence[T], axes: tuple[int, ...]) -> tuple[T, ...]:
    """Select entries of ``values`` corresponding to ``axes``.

    The result is in the order given by ``axes`` (typically still data
    axis order over the selected subset).

    Parameters
    ----------
    values : Sequence[T]
        Per-data-axis sequence (e.g. a shape tuple or scale vector).
    axes : tuple[int, ...]
        Indices into ``values`` to keep.

    Returns
    -------
    tuple[T, ...]
        ``tuple(values[a] for a in axes)``.
    """
    return tuple(values[a] for a in axes)


def swap_axes(values: Sequence[T], permutation: tuple[int, ...]) -> tuple[T, ...]:
    """Reorder ``values`` by an explicit permutation.

    ``permutation[i]`` is the source index whose value becomes output
    index ``i``.  ``permutation`` must be a permutation of
    ``range(len(values))``.

    Parameters
    ----------
    values : Sequence[T]
        Sequence to reorder.
    permutation : tuple[int, ...]
        Permutation of ``range(len(values))``.

    Returns
    -------
    tuple[T, ...]
        ``tuple(values[p] for p in permutation)``.

    Raises
    ------
    ValueError
        If ``permutation`` is not a permutation of ``range(len(values))``.
    """
    n = len(values)
    if len(permutation) != n or sorted(permutation) != list(range(n)):
        raise ValueError(
            f"permutation must be a permutation of range({n}), got {permutation}"
        )
    return tuple(values[p] for p in permutation)


@dataclass(frozen=True)
class RenderSpaces:
    """Everything one visual needs to place its geometry in the scene.

    Rebuilt when ``displayed_axes`` changes -- including a pure reorder, which
    changes the rendered system's axis order and therefore the node matrix,
    while changing nothing about what is fetched (design 3.14).

    Parameters
    ----------
    data : DataCoordinateSystem
        The store's level-0 voxel space, the input of ``data_to_world``.
    data_levels : tuple[DataCoordinateSystem, ...]
        One system per resolution level, finest first, with ``data_levels[0]``
        being ``data``.  A level-2 voxel is not a level-0 voxel, and the
        transform between them is what says so.
    level_transforms : tuple[AffineTransform, ...]
        Level ``k`` voxel space -> level ``0`` voxel space, one per entry in
        ``data_levels``.  Index 0 is the identity.  These are what a region
        is pulled back through, level by level (design 3.11 A), replacing a
        precomputed ``inv_level_k @ inv_visual`` list -- and unlike that list
        they need no inverse at all (D39).
    visual : VisualCoordinateSystem
        The space this visual's GPU geometry is indexed in, for one render
        mode (D45).
    world : WorldCoordinateSystem
        The scene's world.
    rendered : RenderedCoordinateSystem
        The 2D or 3D scene one canvas draws, in **cellier displayed order**
        (Part 5 D1).  The pygfx ``(x, y, z)`` reversal is not carried here; it
        stays at the renderer boundary.
    rendered_to_world : AffineTransform
        The D34 embedding.  Its constant column carries the slice positions.
    world_to_rendered : AffineTransform
        The inverse of the D34 embedding.  Projection onto the displayed
        world axes: it discards the collapsed ones, which the visual has
        already accounted for in ``visual_to_data``.
    retained_axes : tuple[int, ...]
        The **data** axes the visual's geometry keeps, ascending.  Never in
        ``displayed_axes`` order: ``axis_selections`` is assembled per data
        axis ascending and numpy returns an array whose axes are ascending, so
        a display permutation lives in the transform and never in the data.
    data_to_world_axes : Mapping[int, int]
        ``{data axis: world axis}``, the correspondence the visual's transform
        encodes.
    """

    data: DataCoordinateSystem
    visual: VisualCoordinateSystem
    world: WorldCoordinateSystem
    rendered: RenderedCoordinateSystem
    rendered_to_world: AffineTransform
    world_to_rendered: AffineTransform
    retained_axes: tuple[int, ...]
    data_to_world_axes: Mapping[int, int]
    data_levels: tuple[DataCoordinateSystem, ...] = ()
    level_transforms: tuple[AffineTransform, ...] = ()

    @property
    def collapsed_axes(self) -> tuple[int, ...]:
        """The data axes the visual's geometry drops, ascending."""
        retained = set(self.retained_axes)
        return tuple(index for index in range(self.data.ndim) if index not in retained)


def axis_correspondence(transform: BaseTransform) -> dict[int, int]:
    """Read ``{input axis: output axis}`` off a transform.

    A thin delegation to :meth:`BaseTransform.axis_correspondence`, kept as a
    function so this module's callers are unchanged.  The body moved onto the
    transform because the question is structural: a ``ByDimensionTransform``
    answers it from its block declarations, with no matrix, which is what
    lets a non-affine ``data -> world`` transform be asked at all.

    Parameters
    ----------
    transform : BaseTransform
        A ``data -> world`` transform that is axis-aligned -- at most one
        output axis per input axis and vice versa.

    Returns
    -------
    dict[int, int]
        Input axis index to output axis index, for every input axis that
        reaches an output axis.

    Raises
    ------
    ValueError
        If the transform is a shear or a rotation, which the slicing path
        cannot express.
    """
    return transform.axis_correspondence()


def build_render_spaces(
    data_coordinate_system: DataCoordinateSystem,
    visual_coordinate_system: VisualCoordinateSystem,
    world_coordinate_system: WorldCoordinateSystem,
    rendered_coordinate_system: RenderedCoordinateSystem,
    rendered_to_world: AffineTransform,
    data_to_world: AffineTransform,
    retained_axes: Sequence[int],
    data_levels: Sequence[DataCoordinateSystem] = (),
    level_transforms: Sequence[AffineTransform] = (),
) -> RenderSpaces:
    """Assemble a :class:`RenderSpaces` and invert the canvas embedding once.

    Parameters
    ----------
    data_coordinate_system : DataCoordinateSystem
        Level-0 voxel space.
    visual_coordinate_system : VisualCoordinateSystem
        The upload space for this render mode.
    world_coordinate_system : WorldCoordinateSystem
        The scene's world.
    rendered_coordinate_system : RenderedCoordinateSystem
        The canvas's rendered system.
    rendered_to_world : AffineTransform
        The D34 embedding.  Its left inverse is the projection the node matrix
        ends with; an embedding always has one, so this never returns ``None``.
    data_to_world : AffineTransform
        The visual's own transform, read for its axis correspondence.
    retained_axes : Sequence[int]
        The data axes the geometry keeps, ascending.
    data_levels : Sequence[DataCoordinateSystem]
        One system per resolution level, finest first.  Empty for a
        single-level store, where the level-0 system alone says everything.
    level_transforms : Sequence[AffineTransform]
        Level ``k`` -> level ``0``, one per entry in *data_levels*.

    Returns
    -------
    RenderSpaces
        The bundle.

    Raises
    ------
    ValueError
        If the embedding has no inverse, or the transform is sheared.
    """
    world_to_rendered = rendered_to_world.inverse()
    if world_to_rendered is None:
        raise ValueError(
            "The rendered -> world embedding has no left inverse, so there is "
            "no way to express a node matrix in rendered coordinates.  An "
            "embedding built by from_axis_map always has one; this transform "
            "was built some other way."
        )
    return RenderSpaces(
        data=data_coordinate_system,
        visual=visual_coordinate_system,
        world=world_coordinate_system,
        rendered=rendered_coordinate_system,
        rendered_to_world=rendered_to_world,
        world_to_rendered=world_to_rendered,
        retained_axes=tuple(retained_axes),
        data_to_world_axes=axis_correspondence(data_to_world),
        data_levels=tuple(data_levels),
        level_transforms=tuple(level_transforms),
    )


def visual_to_data_transform(
    spaces: RenderSpaces,
    constants: Mapping[int, float],
    scale: Mapping[int, float] | None = None,
    translation: Mapping[int, float] | None = None,
) -> AffineTransform:
    """Build the ``visual -> data`` transform for one request.

    The systems are rebuilt only when ``displayed_axes`` changes; everything
    that varies per request lives here.  ``constants`` is the collapsed voxel
    index per dropped data axis -- the ``int`` entries of ``axis_selections``
    -- and ``translation`` is the window origin, which is zero for a whole
    extent and the brick corner for a windowed one.

    Parameters
    ----------
    spaces : RenderSpaces
        The systems this visual is placed with.
    constants : Mapping[int, float]
        ``{collapsed data axis: voxel index}``.  Every dropped axis must
        appear: an unstated one has no honest default.
    scale : Mapping[int, float] or None
        Per-retained-data-axis scale.  ``None`` is 1 on every axis, which is
        what an array indexed in voxels wants; the multiscale 3D node passes
        the normalized proxy box's factors instead.
    translation : Mapping[int, float] or None
        Per-retained-data-axis translation, keyed by data axis.  ``None`` is
        the origin.

    Returns
    -------
    AffineTransform
        ``visual -> data``, of shape ``(data.ndim + 1, visual.ndim + 1)``.

    Raises
    ------
    ValueError
        If a collapsed axis has no entry in *constants*.
    """
    missing = [axis for axis in spaces.collapsed_axes if axis not in constants]
    if missing:
        raise ValueError(
            f"Collapsed data axes {missing} have no index in constants.  A "
            f"dropped axis sits at a definite voxel and there is no default "
            f"for where."
        )
    data_axes = spaces.data.axes
    visual_axes = spaces.visual.axes
    return AffineTransform.from_axis_map(
        spaces.visual,
        spaces.data,
        axis_map={
            visual_axes[index].id: data_axes[axis].id
            for index, axis in enumerate(spaces.retained_axes)
        },
        scale=(
            None
            if scale is None
            else {
                visual_axes[index].id: scale[axis]
                for index, axis in enumerate(spaces.retained_axes)
                if axis in scale
            }
        ),
        translation=(
            None
            if translation is None
            else {
                visual_axes[index].id: translation[axis]
                for index, axis in enumerate(spaces.retained_axes)
                if axis in translation
            }
        ),
        constant_output_axes={
            data_axes[axis].id: float(constants[axis]) for axis in spaces.collapsed_axes
        },
        name="visual_to_data",
    )


def node_transform(
    spaces: RenderSpaces,
    data_to_world: AffineTransform,
    visual_to_data: AffineTransform,
) -> AffineTransform:
    """Compose ``visual -> rendered``, the transform a node matrix expresses.

    This is design 3.9's chain.  It is square by construction -- the visual
    and rendered systems have the same rank whenever the visual retains
    exactly the displayed axes -- and is handed to ``_pygfx_matrix``, which
    reverses the axis order and narrows to float32 (D6).

    Parameters
    ----------
    spaces : RenderSpaces
        The systems this visual is placed with.
    data_to_world : AffineTransform
        The visual's own transform.
    visual_to_data : AffineTransform
        From :func:`visual_to_data_transform`.

    Returns
    -------
    AffineTransform
        ``visual -> rendered``.
    """
    return visual_to_data.then(data_to_world, spaces.data, spaces.world).then(
        spaces.world_to_rendered, spaces.world, spaces.rendered
    )


def pygfx_to_cellier_order(coordinates: np.ndarray) -> np.ndarray:
    """Reverse the last axis: pygfx ``(x, y, z)`` -> cellier ``(z, y, x)``.

    One of the three places the reversal lives, and the only one that acts on
    **coordinates** rather than on a matrix or a vertex buffer.  Part 5 D1
    keeps the flip at the pygfx boundary rather than folding it into the
    rendered coordinate system, and narrows it to named helpers so that
    moving it later is a change to two functions rather than a hunt for
    ``[[2, 1, 0]]``.

    ``ReslicingRequest.camera_pos`` and ``frustum_corners`` are pygfx
    coordinates and are **not** in the rendered coordinate system.  Nothing in
    the type system says so, which is why they pass through here by name.

    The reversal stays correct when the displayed axes are transposed: it maps
    whatever order the rendered system is in onto pygfx's.

    Parameters
    ----------
    coordinates : np.ndarray
        Points whose last axis is in pygfx order.

    Returns
    -------
    np.ndarray
        The same points with their last axis reversed.
    """
    return np.asarray(coordinates)[..., ::-1]


def cellier_to_pygfx_order(coordinates: np.ndarray) -> np.ndarray:
    """Reverse the last axis: cellier ``(z, y, x)`` -> pygfx ``(x, y, z)``.

    The inverse of :func:`pygfx_to_cellier_order`, and the same operation --
    reversing is its own inverse.  Both names exist so a call site says which
    direction it means.

    Parameters
    ----------
    coordinates : np.ndarray
        Points whose last axis is in cellier order.

    Returns
    -------
    np.ndarray
        The same points with their last axis reversed.
    """
    return np.asarray(coordinates)[..., ::-1]


def pygfx_matrix(transform: AffineTransform) -> np.ndarray:
    """Embed a square 2-D or 3-D transform into a pygfx 4x4 matrix.

    pygfx always requires a 4x4 matrix for ``node.local.matrix``.  This
    function converts from cellier's axis order to pygfx/shader order and
    places the transform into the correct positions of a 4x4 identity matrix.

    Cellier order is ``(z, y, x)`` for 3D and ``(y, x)`` for 2D; pygfx uses
    ``(x, y, z)``, so the axes are reversed.  This is one of the three places
    that reversal lives, and Part 5 D1 keeps it here rather than folding it
    into the rendered coordinate system -- which stays legible against the
    world precisely because it does not carry the flip.

    The reversal is still correct when the displayed axes are **transposed**:
    it maps whatever order the rendered system is in onto pygfx's, so a
    rendered system of ``("X", "Y")`` reverses to pygfx ``(y, x)``.

    This is also one of D6's three narrowing sites: the model layer is float64
    throughout and float32 begins at the GPU boundary.

    Parameters
    ----------
    transform : AffineTransform
        A **square** ``visual -> rendered`` transform of rank 1, 2 or 3.

    Returns
    -------
    np.ndarray
        A ``(4, 4)`` float32 matrix.

    Raises
    ------
    ValueError
        If the transform is not square.  A node matrix maps a visual space
        onto a rendered space of the same rank; an unequal one means the
        visual retained axes the canvas is not showing.
    """
    nd = transform.output_ndim
    if transform.input_ndim != nd:
        raise ValueError(
            f"A node matrix must be square: this transform takes "
            f"{transform.input_ndim} dimensions and produces {nd}.  The "
            f"visual's geometry retains axes the canvas is not displaying."
        )
    src = transform.matrix
    # Reverse axis order: cellier (z, y, x) -> pygfx (x, y, z).
    swap = list(reversed(range(nd)))
    m = np.eye(4, dtype=np.float32)
    for dst_i, src_i in enumerate(swap):
        for dst_j, src_j in enumerate(swap):
            m[dst_i, dst_j] = src[src_i, src_j]
        m[dst_i, 3] = src[src_i, nd]
    return m


def affine_for_node(
    data_to_world: BaseTransform,
    constants: Mapping[int, float],
) -> AffineTransform:
    """Reduce a ``data -> world`` transform to the affine the GPU needs.

    **This is the one structural boundary in the codebase.**  A pygfx node
    transform is a 4x4 matrix, so a non-affine ``data -> world`` has to
    become one here or fail loudly; there is nothing to approximate.

    The question asked is deliberately weaker than "is this transform
    affine".  Every collapsed axis is pinned to *this request's* value
    first, and evaluating any transform at a fixed input gives a constant
    that affine algebra can fold into a translation.  So a non-uniform axis
    that is **sliced** costs nothing here, and only a non-uniform axis that
    is **displayed** raises -- which is the honest outcome, since no matrix
    expresses it.

    An already-affine transform is returned untouched, so nothing about an
    existing scene changes.

    Parameters
    ----------
    data_to_world : BaseTransform
        The visual's own transform.
    constants : Mapping[int, float]
        ``{collapsed data axis: voxel index}`` for this request.

    Returns
    -------
    AffineTransform
        A full-rank ``data -> world`` affine that agrees with
        *data_to_world* wherever the collapsed axes hold their pinned
        values.  The collapsed axes get zero columns: their contribution is
        already a constant in the translation, and the caller feeds them
        those same values.

    Raises
    ------
    NonAffineTransformError
        If what remains after pinning is still not affine -- i.e. a
        non-affine block sits on a displayed axis.  The message names it.
    """
    affine = data_to_world.to_affine()
    if affine is not None:
        return affine

    # restrict() raises NonAffineTransformError naming the axis when a
    # non-affine block is free, which is exactly this boundary's error.
    restricted = data_to_world.restrict(dict(constants)).to_affine()
    if restricted is None:
        raise NonAffineTransformError(
            "This visual's data -> world transform is not affine even with "
            "every collapsed axis pinned, so there is no 4x4 matrix to give "
            "pygfx.  The fix is to stop displaying the non-affine axis."
        )

    input_ndim = data_to_world.input_ndim
    free_axes = [axis for axis in range(input_ndim) if axis not in constants]
    matrix = np.zeros((restricted.matrix.shape[0], input_ndim + 1))
    for column, axis in enumerate(free_axes):
        matrix[:-1, axis] = restricted.linear[:, column]
    matrix[:-1, -1] = restricted.translation
    matrix[-1, -1] = 1.0
    return AffineTransform(
        name=getattr(data_to_world, "name", None),
        input_coordinate_system=data_to_world.input_coordinate_system,
        output_coordinate_system=data_to_world.output_coordinate_system,
        transform=Affine(matrix),
    )


def node_matrix(
    spaces: RenderSpaces,
    data_to_world: BaseTransform,
    constants: Mapping[int, float],
    scale: Mapping[int, float] | None = None,
    translation: Mapping[int, float] | None = None,
) -> np.ndarray:
    """The 4x4 float32 matrix that places one visual's geometry in the scene.

    The whole of design 3.9 in one call: build ``visual -> data`` from this
    request's collapsed indices and window origin, compose it through the
    visual's ``data -> world`` and the canvas's ``world -> rendered``, then
    reverse to pygfx order and narrow to float32.

    Parameters
    ----------
    spaces : RenderSpaces
        The systems this visual is placed with.
    data_to_world : AffineTransform
        The visual's own transform.
    constants : Mapping[int, float]
        ``{collapsed data axis: voxel index}``.
    scale : Mapping[int, float] or None
        Per-retained-data-axis scale of the visual space.  ``None`` is 1,
        which is what an array indexed in voxels wants.
    translation : Mapping[int, float] or None
        Per-retained-data-axis window origin.  ``None`` is the origin.

    Returns
    -------
    np.ndarray
        A ``(4, 4)`` float32 matrix for ``node.local.matrix``.
    """
    return pygfx_matrix(
        node_transform(
            spaces,
            affine_for_node(data_to_world, constants),
            visual_to_data_transform(spaces, constants, scale, translation),
        )
    )


def _extent_along(region: ConvexRegion, index: int) -> float:
    """The region's extent along one half-space's normal.

    Half-spaces come in opposed pairs from every constructor that bounds
    something -- ``from_axis_slabs`` emits ``+e`` / ``-e`` per axis and
    ``from_plane_slab`` emits ``+n`` / ``-n`` -- so the extent is read off the
    pair.  An unpaired half-space bounds one side only and there is nothing to
    widen, so it reports ``inf``.
    """
    normals, offsets = region.normals, region.offsets
    normal = normals[index]
    length = float(np.linalg.norm(normal))
    if length == 0.0:
        return float("inf")
    for other in range(len(normals)):
        if other != index and np.allclose(normals[other], -normal):
            return float(offsets[index] + offsets[other]) / length
    return float("inf")


def with_minimum_thickness(
    region: ConvexRegion, minimum_half_thickness: float
) -> ConvexRegion:
    """Grow a region so it has at least the given half-thickness everywhere.

    A geometry visual draws **points**, which have no extent.  ``contains``
    on a measure-zero region is float-exact and so effectively always
    ``False`` (D42), which means a zero-thickness plane -- what the dims
    editor emits for an axis nobody gave a thickness -- would select nothing
    at all.  So the geometry families give the selection a floor.

    This is a per-family policy on top of the region, not a different
    mechanism: the images want the plane, because they draw one.

    A half-space ``n . p <= d`` is offset outward to ``n . p <= d + s|n|``,
    which is the Minkowski sum with a ball -- exact for an axis slab and for
    an oblique one alike.  A direction the region already has enough extent
    along is left alone, so a thickness the user actually asked for is theirs.

    Parameters
    ----------
    region : ConvexRegion
        The region, in any coordinate system.
    minimum_half_thickness : float
        The floor, in that system's units.

    Returns
    -------
    ConvexRegion
        The region, grown where it was too thin.
    """
    if minimum_half_thickness <= 0.0 or not region.half_spaces:
        return region
    normals, offsets = region.normals, region.offsets
    grown = np.asarray(offsets, dtype=float).copy()
    changed = False
    for index in range(len(normals)):
        extent = _extent_along(region, index)
        shortfall = minimum_half_thickness - extent / 2.0
        if shortfall > 0.0:
            grown[index] += shortfall * float(np.linalg.norm(normals[index]))
            changed = True
    if not changed:
        return region
    return ConvexRegion(
        coordinate_system=region.coordinate_system,
        ndim=region.ndim,
        half_spaces=half_spaces_from_arrays(normals, grown),
    )


def snap_discrete_positions(
    positions: Mapping[int, float],
    data_coordinate_system: DataCoordinateSystem,
    transform: BaseTransform | None = None,
) -> dict[int, float]:
    """Snap positions on **discrete** data axes to the nearest sample.

    **Why a geometry visual needs this at all.**  An image resolves a slice
    position by *snapping to the nearest sample* -- that is what
    ``round_world_to_voxel`` does -- while a geometry visual resolves it by
    *containment in a window*.  Those two flip at different instants: a snap
    flips at the midpoint between two samples, a window flips when the
    position reaches the sample.  With both sharing one world axis, the
    markers visibly lag the image by up to half a sampling interval, and the
    lag is worst exactly where the sampling is coarsest.

    Widening the window cannot fix it: the tolerance required is half the
    *local* gap, which on an irregularly sampled axis is the one thing that
    varies.  Snapping the window's anchor does fix it, exactly, because it
    reuses the same rule -- round half up -- that the image already applies.

    Only axes declared ``sampling="discrete"`` are touched, so a store whose
    time column holds genuine continuous measurements keeps pure containment
    semantics.  A non-finite position is left alone: it means the transform
    reported no preimage, which the caller handles.

    **Rounding is confined to the samples that exist.**  The position handed
    in has already been clamped into the transform's domain by the interval
    semantics of ``imap_bounding_box``, but rounding can push it back out:
    the last sample's cell ends half a unit past its centre, and round-half-up
    sends that boundary *upward*, to a sample one past the end.  Mapping that
    index forward again then has no answer.  So the snapped value is confined
    to the whole samples inside :meth:`BaseTransform.input_domain`.

    This is emphatically **not** "pin an out-of-range position to the edge" --
    that decision was made one layer up and is not revisited here.  It is
    only that rounding must land on a sample that exists.

    Parameters
    ----------
    positions : Mapping[int, float]
        ``{collapsed data axis: position}``, from
        :func:`data_slice_positions`.
    data_coordinate_system : DataCoordinateSystem
        The visual's own data system, which carries the per-axis
        ``sampling``.
    transform : BaseTransform or None
        The visual's ``data -> world`` transform, consulted for the axis's
        domain.  ``None`` skips the confinement, which is right for an axis
        with no intrinsic domain -- an affine one maps every real coordinate,
        so rounding cannot leave anything.

    Returns
    -------
    dict[int, float]
        The same mapping with discrete axes snapped to whole samples.
    """
    domain = {} if transform is None else transform.input_domain()
    snapped = dict(positions)
    axes = data_coordinate_system.axes
    for axis, position in positions.items():
        if not 0 <= axis < len(axes):
            continue
        if axes[axis].sampling != "discrete":
            continue
        if not np.isfinite(position):
            continue
        index = round_half_up(position)
        bounds = domain.get(axis)
        if bounds is not None:
            low, high = bounds
            # The whole samples inside the domain: a span of (-0.5, N - 0.5)
            # holds samples 0 .. N - 1.
            index = max(int(np.ceil(low)), min(index, int(np.floor(high))))
        snapped[axis] = float(index)
    return snapped


def visual_covers_position(
    axis_extents: Sequence[tuple[float, float]] | None,
    data_positions: Mapping[int, float],
) -> bool:
    """Whether a visual has data at the given per-axis **data** positions.

    **Where "out of range means nothing, not the edge sample" is decided.**
    Before this check, a visual whose data ended before the scene's would
    pin its last plane and redraw it forever -- an acquisition ending at 9 s
    re-showing its 9 s frame at 10, 11 and 12 s as though it were data.
    Skipping the request leaves the visual empty instead.

    Both arguments are in the visual's own **data** coordinates, which is
    what keeps this honest: the positions come from
    :func:`data_slice_positions`, which has already pulled the world
    selection back through this visual's own transform, and the extents come
    from its store.  Nothing here maps anything, so there is no opportunity
    to compare a world number against a data one -- the defect this whole
    design exists to prevent.

    Only **collapsed** axes are checked, because those are the only ones
    ``data_slice_positions`` returns: a displayed axis is a range the camera
    looks at, not a position, and a visual partly off-screen still draws the
    part that is on.

    Parameters
    ----------
    axis_extents : Sequence[tuple[float, float]] or None
        The store's per-axis ``(low, high)`` in level-0 data coordinates.
        ``None`` -- an empty store -- covers nothing.
    data_positions : Mapping[int, float]
        ``{collapsed data axis: position}``, from
        :func:`data_slice_positions`.

    Returns
    -------
    bool
        ``False`` when some collapsed position lies outside this visual's
        extent, so the caller should issue no request.
    """
    if axis_extents is None:
        return False
    for axis, position in data_positions.items():
        if not 0 <= axis < len(axis_extents):
            continue
        low, high = axis_extents[axis]
        if not np.isfinite(position):
            # A bounded axis reported no preimage at all for this position.
            return False
        if not (low <= float(position) <= high):
            return False
    return True


def geometry_data_region(
    selection: RegionSelection,
    data_to_world: AffineTransform,
    world: WorldCoordinateSystem,
    minimum_half_thickness: float,
) -> ConvexRegion:
    """The selected region in **data** coordinates, with a thickness floor.

    Design 3.12.  ``imap_region`` is ``A^T`` on the normals and needs no
    inverse (D39), so a geometry visual whose transform is a non-invertible
    embedding still slices -- which is the ``tzyx`` points in a ``TCZYX``
    world of 3.13, where D8 additionally drops the channel constraint the
    dataset has no extent along.

    Parameters
    ----------
    selection : RegionSelection
        The canvas's selection, whose region is in world coordinates.
    data_to_world : AffineTransform
        The visual's own transform.
    world : WorldCoordinateSystem
        Its output system, needed to resolve ``broadcast_axes``.
    minimum_half_thickness : float
        The world-unit floor applied before the pull-back, so that the
        thickness a user states and the thickness a family needs are both
        expressed in the space they were stated in.

    Returns
    -------
    ConvexRegion
        The region in the visual's data coordinates.
    """
    widened = with_minimum_thickness(selection.region, minimum_half_thickness)
    return data_to_world.imap_region(widened, world).simplify()


def axis_scales(data_to_world: AffineTransform) -> dict[int, float]:
    """World units per data unit, keyed by **data** axis.

    Read off the single entry each data axis contributes, which is what an
    axis-aligned transform has (see :func:`axis_correspondence`).  Used to
    convert a thickness stated in world units into the data units a store's
    own slab arithmetic works in.

    **Stays `AffineTransform`-only, deliberately.**  On a non-uniform axis
    there is no single world-units-per-data-unit -- the spacing is what
    varies -- so this function has no answer to give and must not pretend to
    by returning some average.  A caller that needs to convert a world
    window on a possibly-non-affine axis pulls the window's two **endpoints**
    back through the transform instead of dividing its width; that is exact
    for a monotonic axis and algebraically identical to dividing by the
    scale on an affine one.  See ``render/visuals/_graph_memory.py``, which
    is the only caller that ever needed it.

    Parameters
    ----------
    data_to_world : AffineTransform
        The visual's transform.  Affine specifically, not ``BaseTransform``.

    Returns
    -------
    dict[int, float]
        Data axis to the magnitude of its scale.  Axes that reach no world
        axis are absent.
    """
    linear = np.asarray(data_to_world.linear)
    return {
        data_axis: abs(float(linear[world_axis, data_axis]))
        for data_axis, world_axis in axis_correspondence(data_to_world).items()
    }


def data_slice_positions(
    region: ConvexRegion,
    data_to_world: AffineTransform,
    world: WorldCoordinateSystem,
) -> dict[int, float]:
    """Where the selection sits, per collapsed data axis, in data units.

    For the families whose own slab arithmetic the region does not replace --
    the graph, whose trail is an asymmetric window with a fade measured from
    the slice (design 3.12's "per-family policy on top of the region").  They
    need the position, not the extent, so this returns the centre of the
    pulled-back region on every axis it bounds.

    Parameters
    ----------
    region : ConvexRegion
        The selection's region, in world coordinates.
    data_to_world : AffineTransform
        The visual's transform.
    world : WorldCoordinateSystem
        Its output system, needed to resolve ``broadcast_axes``.

    Returns
    -------
    dict[int, float]
        Data axis to slice position.  Axes the region leaves unbounded are
        absent.

    Raises
    ------
    ValueError
        If an axis comes back bounded on one side only, which is the shear
        case D7 rejects.
    """
    box = data_to_world.imap_region(region, world).simplify().bounding_box()
    positions: dict[int, float] = {}
    for axis in range(box.ndim):
        low = float(box.min_coordinate[axis])
        high = float(box.max_coordinate[axis])
        low_unbounded, high_unbounded = np.isneginf(low), np.isposinf(high)
        if low_unbounded and high_unbounded:
            continue
        if low_unbounded or high_unbounded:
            raise ValueError(
                f"Data axis {axis} of this region is bounded on one side only "
                f"([{low}, {high}]).  That is a cross-term on a collapsed "
                f"axis; remove the shear."
            )
        positions[axis] = (low + high) / 2.0
    return positions
