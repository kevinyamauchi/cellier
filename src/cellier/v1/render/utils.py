"""Utilities for interfacing between cellier models and PyGFX objects."""

from cellier.models.nodes.base_node import BaseNode
from cellier.models.nodes.image_node import ImageNode, MultiscaleImageNode
from cellier.models.nodes.mesh_node import MeshNode
from cellier.models.nodes.points_node import PointsNode
from cellier.render.image import GFXImageNode, GFXMultiScaleImageNode
from cellier.render.mesh import GFXMeshNode
from cellier.render.points import GFXPointsVisual


def construct_pygfx_object(node_model: BaseNode):
    """Construct a PyGFX object from a cellier visual model."""
    if isinstance(node_model, MeshNode):
        # mesh
        return GFXMeshNode(node_model)

    elif isinstance(node_model, PointsNode):
        # points
        return GFXPointsVisual(model=node_model)
    elif isinstance(node_model, ImageNode):
        return GFXImageNode(model=node_model)
    elif isinstance(node_model, MultiscaleImageNode):
        return GFXMultiScaleImageNode(model=node_model)

    else:
        raise TypeError(f"Unsupported visual model: {type(node_model)}")
