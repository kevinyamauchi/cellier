"""The transform base class (design section 8)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import UUID4, BaseModel, ConfigDict, Field

# Transform is imported at runtime, not under TYPE_CHECKING: pydantic
# resolves field annotations at class-creation time and cannot see a
# deferred import.
from transformnd.base import Transform  # noqa: TC002

from cellier.transform._geometry_ops import (
    DegenerateNormalError,
    NonInvertibleTransformError,
)

if TYPE_CHECKING:
    import numpy as np

    from cellier.transform._coordinate_system import CoordinateSystem
    from cellier.transform._geometry import AxisAlignedBoundingBox, Plane
    from cellier.transform._region import ConvexRegion

__all__ = [
    "BaseTransform",
    "DegenerateNormalError",
    "NonInvertibleTransformError",
]


class BaseTransform(BaseModel, ABC):
    """A transform between two coordinate systems.

    Wraps a :class:`transformnd.base.Transform` rather than
    reimplementing one.  ``transformnd.Spaced`` is deliberately not used:
    it carries no axis metadata, does not serialize, and its ``invert()``
    returns the original space order while performing the inverted
    transform.  Owning the source/target pairing here is what avoids
    inheriting that bug.

    Transforms are frozen (Q13).  Dimensionality is **not** stored: it
    lives on the wrapped transform and is read through, so the model
    cannot disagree with itself (D9).  The cross-check against the
    coordinate systems' axis counts cannot happen here, because the model
    holds ids rather than systems; it is factored into
    :meth:`validate_against`.

    Parameters
    ----------
    name : str or None
        Optional human-readable name.  RFC-5 gives transformations one.
    input_coordinate_system : UUID4
        The id of the coordinate system this transform maps from.
    output_coordinate_system : UUID4
        The id of the coordinate system this transform maps to.
    transform : Transform
        The wrapped ``transformnd`` transform.
    id : UUID4
        Unique identifier.  Auto-generated.  Not ``(input, output)``,
        because two coordinate systems may be joined by more than one
        transform (D14).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str | None = None
    input_coordinate_system: UUID4
    output_coordinate_system: UUID4
    transform: Transform
    id: UUID4 = Field(default_factory=uuid4, frozen=True)

    # -- dimensionality, read through to the wrapped transform (D9) ----

    @property
    def input_ndim(self) -> int:
        """Number of input dimensions, read from the wrapped transform."""
        return self.transform.ndims.source

    @property
    def output_ndim(self) -> int:
        """Number of output dimensions, read from the wrapped transform."""
        return self.transform.ndims.target

    # -- validation ----------------------------------------------------

    def validate_against(
        self,
        input_coordinate_system: CoordinateSystem,
        output_coordinate_system: CoordinateSystem,
    ) -> None:
        """Check this transform against the two systems it claims to join.

        The four checks of D9: both ids match, and both axis counts match
        the wrapped transform's dimensionality.  A future registry calls
        this on insertion; defining it now means the registry inherits a
        decided rule rather than inventing one.

        Parameters
        ----------
        input_coordinate_system : CoordinateSystem
            The system this transform should map from.
        output_coordinate_system : CoordinateSystem
            The system this transform should map to.

        Raises
        ------
        ValueError
            If any of the four checks fails.
        """
        if input_coordinate_system.id != self.input_coordinate_system:
            raise ValueError(
                f"Input coordinate system id mismatch: transform expects "
                f"{self.input_coordinate_system}, got "
                f"{input_coordinate_system.id}."
            )
        if output_coordinate_system.id != self.output_coordinate_system:
            raise ValueError(
                f"Output coordinate system id mismatch: transform expects "
                f"{self.output_coordinate_system}, got "
                f"{output_coordinate_system.id}."
            )
        if input_coordinate_system.ndim != self.input_ndim:
            raise ValueError(
                f"Input rank mismatch: transform takes {self.input_ndim} "
                f"dimensions, coordinate system "
                f"'{input_coordinate_system.name}' has "
                f"{input_coordinate_system.ndim} axes."
            )
        if output_coordinate_system.ndim != self.output_ndim:
            raise ValueError(
                f"Output rank mismatch: transform produces "
                f"{self.output_ndim} dimensions, coordinate system "
                f"'{output_coordinate_system.name}' has "
                f"{output_coordinate_system.ndim} axes."
            )

    def _check_input_coordinate_system(self, coordinate_system: UUID4) -> None:
        """Raise if an argument is not in this transform's input system.

        Parameters
        ----------
        coordinate_system : UUID4
            The id carried by the argument.

        Raises
        ------
        ValueError
            If it is not the input coordinate system.
        """
        if coordinate_system != self.input_coordinate_system:
            raise ValueError(
                f"Expected an object in coordinate system "
                f"{self.input_coordinate_system} (this transform's input), "
                f"got one in {coordinate_system}."
            )

    def _check_output_coordinate_system(self, coordinate_system: UUID4) -> None:
        """Raise if an argument is not in this transform's output system.

        Parameters
        ----------
        coordinate_system : UUID4
            The id carried by the argument.

        Raises
        ------
        ValueError
            If it is not the output coordinate system.
        """
        if coordinate_system != self.output_coordinate_system:
            raise ValueError(
                f"Expected an object in coordinate system "
                f"{self.output_coordinate_system} (this transform's output), "
                f"got one in {coordinate_system}."
            )

    # -- points --------------------------------------------------------

    @abstractmethod
    def map_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Map points from the input system to the output system."""

    @abstractmethod
    def imap_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Map points from the output system back to the input system."""

    # -- vectors, two kinds and never one (D27) ------------------------

    @abstractmethod
    def map_direction(self, direction: np.ndarray) -> np.ndarray:
        """Map displacement vectors forward (contravariant)."""

    @abstractmethod
    def imap_direction(self, direction: np.ndarray) -> np.ndarray:
        """Map displacement vectors back (contravariant)."""

    @abstractmethod
    def map_normal(self, normal: np.ndarray) -> np.ndarray:
        """Map plane normals forward (covariant)."""

    @abstractmethod
    def imap_normal(self, normal: np.ndarray) -> np.ndarray:
        """Map plane normals back (covariant)."""

    # -- regions -------------------------------------------------------

    @abstractmethod
    def map_bounding_box(
        self,
        box: AxisAlignedBoundingBox,
        output_coordinate_system: CoordinateSystem,
    ) -> AxisAlignedBoundingBox:
        """Map an axis-aligned box forward, conservatively."""

    @abstractmethod
    def imap_bounding_box(self, box: AxisAlignedBoundingBox) -> AxisAlignedBoundingBox:
        """Map an axis-aligned box back, conservatively."""

    @abstractmethod
    def map_plane(self, plane: Plane) -> Plane:
        """Map a plane forward."""

    @abstractmethod
    def imap_plane(self, plane: Plane) -> Plane:
        """Map a plane back."""

    @abstractmethod
    def map_region(self, region: ConvexRegion) -> ConvexRegion:
        """Map a convex region forward."""

    @abstractmethod
    def imap_region(
        self,
        region: ConvexRegion,
        output_coordinate_system: CoordinateSystem,
    ) -> ConvexRegion:
        """Map a convex region back, dropping broadcast constraints (D8)."""

    # -- inversion -----------------------------------------------------

    @abstractmethod
    def inverse(self) -> BaseTransform | None:
        """Return the inverse transform, or ``None`` if there is none."""
