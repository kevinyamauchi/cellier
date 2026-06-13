"""Utilities for interfacing between cellier models and PyGFX objects."""

from cellier.legacy.models.visuals import (
    LinesVisual,
    MultiscaleImageVisual,
    MultiscaleLabelsVisual,
    PointsVisual,
)
from cellier.legacy.models.visuals.base import BaseVisual
from cellier.legacy.render.image import GFXMultiScaleImageNode
from cellier.legacy.render.labels import GFXMultiScaleLabelsNode
from cellier.legacy.render.lines import GFXLinesVisual
from cellier.legacy.render.points import GFXPointsVisual


def construct_pygfx_object(visual_model: BaseVisual):
    """Construct a PyGFX object from a cellier visual model."""
    if isinstance(visual_model, PointsVisual):
        # points
        return GFXPointsVisual(model=visual_model)

    elif isinstance(visual_model, LinesVisual):
        # lines
        return GFXLinesVisual(model=visual_model)

    elif isinstance(visual_model, MultiscaleLabelsVisual):
        # labels
        return GFXMultiScaleLabelsNode(model=visual_model)
    elif isinstance(visual_model, MultiscaleImageVisual):
        # images
        return GFXMultiScaleImageNode(model=visual_model)
    else:
        raise TypeError(f"Unsupported visual model: {type(visual_model)}")
