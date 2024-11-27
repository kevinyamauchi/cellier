"""Utilities for interfacing between cellier models and PyGFX objects."""

from cellier.models.data_manager import DataManager
from cellier.models.nodes.base_node import BaseNode
from cellier.models.nodes.image_node import ImageNode
from cellier.models.nodes.mesh_node import MeshNode
from cellier.models.nodes.points_node import PointsNode
from cellier.render.image import GFXImageNode
from cellier.render.mesh import GFXMeshNode
from cellier.render.points import GFXPointsNode


def construct_pygfx_object(node_model: BaseNode, data_manager: DataManager):
    """Construct a PyGFX object from a cellier visual model."""
    if isinstance(node_model, MeshNode):
        # mesh
        return GFXMeshNode(data_manager, node_model)

    elif isinstance(node_model, PointsNode):
        # points
        return GFXPointsNode(model=node_model)
    elif isinstance(node_model, ImageNode):
        return GFXImageNode(model=node_model)

    else:
        raise TypeError(f"Unsupported visual model: {type(node_model)}")
