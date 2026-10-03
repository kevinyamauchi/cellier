"""Clipping planes in the render layer (clipping planes design v2, 4.1, 4.6).

A visual's planes are authored in its level-0 data coordinates and may
have a component on any data axis.  ``reduce_clipping_planes`` turns them
into what one view tests against: ``(a, b, c, d)`` in the scene's rendered
space, in pygfx order, kept where ``a*x + b*y + c*z >= d``.  In a 2D view
the depth component is zero and the result is the line where the plane
meets the slice.

``ClippingPlanesMixin`` gives a render visual the one seam through which
planes reach its materials.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from cellier.data._plane_clip import PlaneTuple, plane_tuples
from cellier.render._spaces import affine_for_node

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from cellier.render._spaces import RenderSpaces
    from cellier.transform import BaseTransform

#: What a disabled plane is uploaded as.  It keeps everything
#: (``0 >= -1``), so toggling ``enabled`` never changes the plane count and
#: never recompiles a shader.
KEEP_EVERYTHING: tuple[float, float, float, float] = (0.0, 0.0, 0.0, -1.0)


def reduce_clipping_planes(
    spaces: RenderSpaces,
    data_to_world: BaseTransform,
    constants: Mapping[int, float],
    planes: Sequence[Any],
) -> list[tuple[float, float, float, float]]:
    """Reduce data-space clipping planes to one view's rendered space.

    Takes the same ``constants`` as :func:`~cellier.render._spaces.node_matrix`
    and must be given the same values: the plane is cut by the slice the
    node is drawn at.

    For a plane ``n . p >= d``: the collapsed axes are substituted
    (``d_eff = d - sum(n[h] * constants[h])``), and the rest is carried
    through ``r = L x + tau``, the retained data axes to rendered space:
    ``n' = L^-T n_retained``, ``d' = d_eff + n' . tau``.  It never calls
    ``transform.map_plane``, so a non-affine collapsed axis (a non-uniform
    time axis) is fine.

    Parameters
    ----------
    spaces : RenderSpaces
        The systems the visual is placed with.
    data_to_world : BaseTransform
        The visual's own transform.
    constants : Mapping[int, float]
        ``{collapsed data axis: data position}``.
    planes : Sequence[ClippingPlane]
        The visual's planes, in level-0 data coordinates.

    Returns
    -------
    list[tuple[float, float, float, float]]
        One ``(a, b, c, d)`` per plane, in pygfx ``(x, y, z)`` order.  A
        disabled plane gives :data:`KEEP_EVERYTHING`.  A plane parallel to
        the slice gives ``(0, 0, 0, d')``: everything or nothing.
    """
    if not planes:
        return []
    rendered = affine_for_node(data_to_world, constants).then(
        spaces.world_to_rendered, spaces.world, spaces.rendered
    )
    ndim = rendered.linear.shape[1]
    retained = [axis for axis in range(ndim) if axis not in constants]
    linear = np.asarray(rendered.linear, dtype=np.float64)[:, retained]
    tau = np.asarray(rendered.translation, dtype=np.float64)
    out: list[tuple[float, float, float, float]] = []
    for item in planes:
        if not item.enabled:
            out.append(KEEP_EVERYTHING)
            continue
        normal = item.plane.normal
        offset = item.plane.offset - sum(
            float(normal[axis]) * float(value) for axis, value in constants.items()
        )
        reduced = np.linalg.solve(linear.T, normal[retained])
        abc = np.zeros(3, dtype=np.float64)
        abc[: len(reduced)] = reduced[::-1]
        out.append(
            (float(abc[0]), float(abc[1]), float(abc[2]), offset + float(reduced @ tau))
        )
    return out


def data_half_space_rows(
    planes: Sequence[Any],
    retained_axes: Sequence[int],
    constants: Mapping[int, float],
) -> np.ndarray:
    """The enabled planes as culling rows over the retained data axes.

    Rows are ``(n_x, n_y, n_z, w)`` (2D: the depth entry is zero) in level-0
    data coordinates of the retained axes in pygfx order, kept where
    ``dot(row[:3], p) + row[3] >= 0``: the form the frustum rows of brick
    and tile culling take, so the two concatenate.

    Parameters
    ----------
    planes : Sequence[ClippingPlane]
        The visual's planes.
    retained_axes : Sequence[int]
        The data axes the view keeps, ascending.
    constants : Mapping[int, float]
        ``{collapsed data axis: data position}``.

    Returns
    -------
    np.ndarray
        ``(n_enabled, 4)`` float64.
    """
    rows = []
    retained = list(retained_axes)
    for item in planes:
        if not item.enabled:
            continue
        normal = item.plane.normal
        offset = item.plane.offset - sum(
            float(normal[axis]) * float(value) for axis, value in constants.items()
        )
        row = np.zeros(4, dtype=np.float64)
        kept = normal[retained][::-1]
        row[: len(kept)] = kept
        row[3] = -offset
        rows.append(row)
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4)


def drawn_materials(*nodes: Any) -> list[Any]:
    """The materials of every image and volume under *nodes*.

    Lines are left out: the bounding-box wireframe sits under the same
    nodes and is not clipped.
    """
    import pygfx as gfx

    found = []
    for node in nodes:
        if node is None:
            continue
        for obj in node.iter(lambda o: isinstance(o, (gfx.Volume, gfx.Image))):
            material = getattr(obj, "material", None)
            if material is not None:
                found.append(material)
    return found


class ClippingPlanesMixin:
    """The one way clipping planes reach a render visual's materials.

    The planes are state of the render visual, not of a material: a
    material that is created or swapped in later must be given them.  A
    host class:

    - implements :meth:`_clip_targets`;
    - calls :meth:`_apply_clipping_planes` wherever it places its node (the
      slice the node is drawn at may have moved) and wherever it creates or
      assigns a material.

    It reads ``self._spaces`` and ``self._transform``.
    """

    _clipping_planes: tuple = ()

    #: Whether a change of planes changes what this visual reads, so the
    #: controller reslices it.  ``True`` for multiscale image and labels
    #: (clipped bricks and tiles are not fetched).
    clipping_planes_affect_request: bool = False

    @property
    def clipping_planes(self) -> tuple:
        """The planes last given to this visual."""
        return self._clipping_planes

    def set_clipping_planes(self, planes: Iterable[Any]) -> None:
        """Adopt *planes* and write them to every material."""
        self._clipping_planes = tuple(planes)
        self._apply_clipping_planes()

    def on_clipping_planes_changed(self, event: Any) -> None:
        """Bus handler for ``ClippingPlanesChangedEvent``."""
        self.set_clipping_planes(event.clipping_planes)

    def _clip_targets(self) -> Iterable[tuple[Iterable[Any], Mapping[int, float]]]:
        """Yield ``(materials, constants)`` groups.

        ``constants`` are the collapsed data positions the group's node is
        drawn at.  Materials held aside (an empty placeholder, the real
        material while the placeholder is shown) are included, so one that
        comes back does not bring stale planes.
        """
        raise NotImplementedError

    def _apply_clipping_planes(self) -> None:
        """Reduce the stored planes and write them to every material."""
        spaces = getattr(self, "_spaces", None)
        transform = getattr(self, "_transform", None)
        planes = self._clipping_planes
        if getattr(self, "_clip_on_cpu", False):
            # The read clips; a reduced plane would cut the flattened
            # geometry a second time, at the wrong place.
            planes = ()
        placed = spaces is not None and transform is not None
        for materials, constants in self._clip_targets():
            if not planes:
                reduced: list = []
            elif not placed:
                continue
            else:
                reduced = reduce_clipping_planes(spaces, transform, constants, planes)
            for material in materials:
                if material is None:
                    continue
                if not reduced and not material.clipping_planes:
                    continue
                material.clipping_planes = reduced


class GeometryClippingMixin(ClippingPlanesMixin):
    """Clipping for geometry that a view may flatten (design 4.2, 5.2).

    The shader is exact whenever the rendered position determines the
    position on every axis a plane has a component on.  Geometry flattened
    along such an axis (a 2D view of a slab, a trail along time) has lost
    that coordinate, so the read clips it instead, by its true position,
    and the shader does not.

    A host calls :meth:`_begin_request_clipping` when it builds a request.
    """

    #: Whether the last request was clipped in the read.
    _clip_on_cpu: bool = False

    def _wants_cpu_clip(self) -> bool:
        """Whether an enabled plane has a component on a collapsed axis."""
        spaces = getattr(self, "_spaces", None)
        if spaces is None or not self._clipping_planes:
            return False
        collapsed = list(spaces.collapsed_axes)
        if not collapsed:
            return False
        return any(
            item.enabled and bool(np.any(item.plane.normal[collapsed] != 0.0))
            for item in self._clipping_planes
        )

    @property
    def clipping_planes_affect_request(self) -> bool:
        """A change of planes needs a read when the read clips, or did."""
        return self._clip_on_cpu or self._wants_cpu_clip()

    def _begin_request_clipping(self) -> tuple[PlaneTuple, ...]:
        """Decide where this request is clipped, and tell the materials.

        Returns
        -------
        tuple[PlaneTuple, ...]
            The planes the read applies, as ``(normal, offset)`` pairs in
            data coordinates; empty when the shader clips.
        """
        self._clip_on_cpu = self._wants_cpu_clip()
        self._apply_clipping_planes()
        return plane_tuples(self._clipping_planes) if self._clip_on_cpu else ()
