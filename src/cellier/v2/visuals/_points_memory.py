# src/cellier/v2/visuals/_points_memory.py
from __future__ import annotations

from typing import Literal

from pydantic import Field

from cellier.v2.visuals._base_visual import BaseAppearance, BaseVisual


class PointsMarkerAppearance(BaseAppearance):
    """Appearance model for a points visual.

    Parameters
    ----------
    color : tuple[float, float, float, float]
        RGBA uniform fallback colour.  Used when ``color_mode`` is
        ``"uniform"`` or when no per-point colors are present.
    size : float
        Uniform point size, interpreted in ``size_space`` units.
    size_space : str
        Coordinate space for size.  ``"screen"`` (pixels) or
        ``"world"`` (world-space units).  Default ``"screen"``.
    color_mode : str
        ``"uniform"`` — use ``color`` for every point.
        ``"vertex"`` — use per-point colors from the geometry buffer.
    opacity : float
        Master opacity multiplier in [0, 1].
    """

    appearance_type: Literal["points_marker"] = "points_marker"
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    size: float = 5.0
    size_space: Literal["screen", "world"] = "screen"
    color_mode: Literal["uniform", "vertex"] = "uniform"


class PointsVisual(BaseVisual):
    """Model-layer visual for a point cloud backed by PointsMemoryStore.

    Parameters
    ----------
    name : str
        Human-readable label.
    data_store_id : str
        UUID of the backing PointsMemoryStore, as a string.
    appearance : PointsMarkerAppearance
        Initial appearance.
    requires_camera_reslice : bool
        Always False — points do not depend on camera position.
    """

    visual_type: Literal["points"] = "points"
    name: str
    data_store_id: str
    appearance: PointsMarkerAppearance = Field(default_factory=PointsMarkerAppearance)
    requires_camera_reslice: bool = False
