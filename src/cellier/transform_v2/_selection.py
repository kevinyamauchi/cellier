"""One canvas's view of the world: where the slice is, and how thick (D43)."""

from __future__ import annotations

import numpy as np  # noqa: TC002
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

# Both models are imported at runtime, not under TYPE_CHECKING: pydantic
# resolves field annotations at class-creation time and cannot see a
# deferred import.
from cellier.transform_v2._affine import AffineTransform  # noqa: TC001
from cellier.transform_v2._region import ConvexRegion  # noqa: TC001


class RegionSelection(BaseModel):
    """A slice position and its extent, as one frozen artifact.

    This is the authoritative artifact the slicer consumes.  A
    ``DimsManager`` becomes an *editor* that holds the ergonomic
    parameterization -- an index and a thickness, the thing a GUI binds
    to -- and emits one of these; an oblique editor holds a plane and a
    thickness and emits the same type.  The slicer consumes one thing and
    does not know which editor produced it (R6).

    The numbers stay in one place.  The transform's constant column holds
    the slice **position** and the region holds its **extent**.  Both are
    downstream of the editor's index, and neither is stored twice.

    The region is in **world** coordinates, not rendered.  Rendered space
    has only the two or three displayed axes and so cannot express a
    thickness on a collapsed one; world space expresses both, and
    obliquity is carried by the half-space normals rather than by another
    coordinate system.

    Parameters
    ----------
    transform : AffineTransform
        The rendered -> world embedding (D34).  Its translation column
        carries the slice indices.
    region : ConvexRegion
        The selected region, in the transform's **output** (world)
        coordinate system.
    """

    model_config = ConfigDict(frozen=True)

    transform: AffineTransform
    region: ConvexRegion

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        """Check that the slice the transform names selects something.

        Three checks.  The region must be in the transform's output
        system and of matching rank -- otherwise the two halves describe
        different spaces.  And the invariant that actually connects them:
        **some rendered point lands inside the region**, i.e. pulling the
        region back through the transform leaves a non-empty set.  A
        selection whose slice sits outside its own region is incoherent,
        and this is what catches it.

        D43 states that invariant as "the transform's translation
        satisfies every half-space".  That is the same test whenever the
        region constrains only *collapsed* axes, because the translation
        is exactly the slice position on those.  It is too strong as
        soon as a **displayed** axis is bounded, which R3 explicitly
        permits: the translation is zero on a displayed axis, so
        ``Z in [11, 13]`` on a displayed ``Z`` would be rejected even
        though every rendered point with ``Z in [11, 13]`` is in the
        region.  The pull-back accepts exactly the selections that select
        something and still rejects the case D43 names.

        This is a consistency check on a hand-built selection, not a
        synchronisation mechanism.  Nothing here updates one half when
        the other changes; that is the editor's job.

        Returns
        -------
        RegionSelection
            The validated selection.

        Raises
        ------
        ValueError
            If the region is in the wrong coordinate system, has the
            wrong rank, declares broadcast axes, or excludes every point
            the transform can reach.
        """
        if self.region.coordinate_system != self.transform.output_coordinate_system:
            raise ValueError(
                f"The region must be in the transform's output coordinate "
                f"system {self.transform.output_coordinate_system}, but it is "
                f"in {self.region.coordinate_system}."
            )
        if self.region.ndim != self.transform.output_ndim:
            raise ValueError(
                f"The region has rank {self.region.ndim} but the transform "
                f"produces {self.transform.output_ndim} dimensions."
            )
        if self.transform.broadcast_axes:
            raise ValueError(
                "A RegionSelection's transform is the rendered -> world "
                "embedding of D34, which records where the slice sits as "
                "constant_output_axes.  It must not declare broadcast_axes: a "
                "world axis the canvas does not display is at a definite "
                "position, not free.  Rebuild the embedding with "
                "constant_output_axes instead of broadcast_output_axes."
            )
        # The embedding has no broadcast axes -- checked immediately above --
        # so there is nothing for D8's drop rule to remove and the indices
        # are empty.  This is the one caller that has the ids but not the
        # world CoordinateSystem object needed to resolve them.
        if self.transform._imap_region(self.region, ()).is_empty():
            raise ValueError(
                f"No point this transform can reach lies inside the selected "
                f"region: the selection's own slice position "
                f"{self.transform.translation.tolist()} would select nothing."
            )
        return self

    @property
    def slice_position(self) -> np.ndarray:
        """The world point the rendered origin maps to.

        This is the transform's translation column, which is where the
        slice indices live (D35).
        """
        return self.transform.translation
