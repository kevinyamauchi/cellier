# src/cellier/v2/visuals/_lines_memory.py
from __future__ import annotations

from typing import Literal

from pydantic import Field

from cellier.v2.visuals._base_visual import BaseAppearance, BaseVisual


class LinesMemoryAppearance(BaseAppearance):
    """Appearance model for a lines visual.

    Parameters
    ----------
    color : tuple[float, float, float, float]
        RGBA uniform fallback colour.  Used when ``color_mode`` is
        ``"uniform"`` or when no per-vertex colors are present.
    thickness : float
        Line thickness, interpreted in ``thickness_space`` units.
    thickness_space : str
        Coordinate space for thickness.  ``"screen"`` (logical pixels)
        or ``"world"`` (world-space units).  Default ``"screen"``.
    color_mode : str
        ``"uniform"`` — use ``color`` for every vertex.
        ``"vertex"`` — use per-vertex colors from the geometry buffer.
    opacity : float
        Master opacity multiplier in [0, 1].
    """

    appearance_type: Literal["lines_memory"] = "lines_memory"
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    thickness: float = 2.0
    thickness_space: Literal["screen", "world"] = "screen"
    color_mode: Literal["uniform", "vertex"] = "uniform"
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    depth_test: bool = True


class LinesVisual(BaseVisual):
    """Model-layer visual for a line-segment collection backed by LinesMemoryStore.

    Parameters
    ----------
    name : str
        Human-readable label.
    data_store_id : str
        UUID of the backing LinesMemoryStore, as a string.
    appearance : LinesMemoryAppearance
        Initial appearance.
    requires_camera_reslice : bool
        Always False — lines do not depend on camera position.
    """

    visual_type: Literal["lines"] = "lines"
    name: str
    data_store_id: str
    appearance: LinesMemoryAppearance = Field(default_factory=LinesMemoryAppearance)
    requires_camera_reslice: bool = False
