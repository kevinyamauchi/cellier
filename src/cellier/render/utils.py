"""Utilities for interfacing between cellier models and PyGFX objects."""

from cellier.models.visuals import LinesVisual, PointsVisual
from cellier.models.visuals.base import BaseVisual
from cellier.render.lines import GFXLinesVisual
from cellier.render.points import GFXPointsVisual


def construct_pygfx_object(node_model: BaseVisual):
    """Construct a PyGFX object from a cellier visual model."""
    if isinstance(node_model, PointsVisual):
        # points
        return GFXPointsVisual(model=node_model)

    elif isinstance(node_model, LinesVisual):
        # lines
        return GFXLinesVisual(model=node_model)

    else:
        raise TypeError(f"Unsupported visual model: {type(node_model)}")
