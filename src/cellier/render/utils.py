"""Utilities for interfacing between cellier models and PyGFX objects."""

from cellier.models.data_manager import DataManager
from cellier.models.visuals.base_visual import BaseVisual
from cellier.models.visuals.mesh_visual import MeshVisual
from cellier.models.visuals.points_visual import PointsVisual
from cellier.render.mesh import construct_pygfx_mesh_from_model
from cellier.render.points import GFXPointsNode


def construct_pygfx_object(node_model: BaseVisual, data_manager: DataManager):
    """Construct a PyGFX object from a cellier visual model."""
    if isinstance(node_model, MeshVisual):
        # mesh
        return construct_pygfx_mesh_from_model(
            model=node_model, data_manager=data_manager
        )
    elif isinstance(node_model, PointsVisual):
        # points
        return GFXPointsNode(model=node_model)
    else:
        raise TypeError(f"Unsupported visual model: {type(node_model)}")
