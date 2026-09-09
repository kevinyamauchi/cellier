"""Coordinate systems and transforms (v2).

This package is independent of :mod:`cellier.transform`: neither imports
from the other (D1).  Nothing here is wired into the viewer yet, so it is
deliberately not re-exported from :mod:`cellier`.
"""

from cellier.transform_v2._affine import AffineTransform
from cellier.transform_v2._axis import Axis, AxisRef, AxisType
from cellier.transform_v2._base import BaseTransform
from cellier.transform_v2._coordinate_system import (
    CoordinateSystem,
    CoordinateSystemType,
    DataCoordinateSystem,
    RenderedCoordinateSystem,
    WorldCoordinateSystem,
)
from cellier.transform_v2._geometry import AxisAlignedBoundingBox, Plane
from cellier.transform_v2._geometry_ops import (
    DegenerateNormalError,
    NonInvertibleTransformError,
)
from cellier.transform_v2._region import ConvexRegion, HalfSpace
from cellier.transform_v2._selection import RegionSelection

__all__ = [
    "AffineTransform",
    "Axis",
    "AxisAlignedBoundingBox",
    "AxisRef",
    "AxisType",
    "BaseTransform",
    "ConvexRegion",
    "CoordinateSystem",
    "CoordinateSystemType",
    "DataCoordinateSystem",
    "DegenerateNormalError",
    "HalfSpace",
    "NonInvertibleTransformError",
    "Plane",
    "RegionSelection",
    "RenderedCoordinateSystem",
    "WorldCoordinateSystem",
]
