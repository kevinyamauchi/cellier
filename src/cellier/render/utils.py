"""Utilities for interfacing between cellier models and PyGFX objects."""

from cellier.models.nodes.base_node import BaseNode
from cellier.models.nodes.lines_node import LinesNode
from cellier.models.nodes.points_node import PointsNode
from cellier.render.lines import GFXLinesNode
from cellier.render.points import GFXPointsNode


def construct_pygfx_object(node_model: BaseNode):
    """Construct a PyGFX object from a cellier visual model."""
    if isinstance(node_model, PointsNode):
        # points
        return GFXPointsNode(model=node_model)

    elif isinstance(node_model, LinesNode):
        # lines
        return GFXLinesNode(model=node_model)

    else:
        raise TypeError(f"Unsupported visual model: {type(node_model)}")
