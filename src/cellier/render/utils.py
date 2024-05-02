"""Utilities for interfacing between cellier models and PyGFX objects."""

from cellier.models.data_manager import DataManager
from cellier.models.visuals.base_visual import BaseVisual
from cellier.models.visuals.mesh_visual import MeshVisual
from cellier.models.visuals.points_visual import PointsVisual
from cellier.render.mesh import construct_pygfx_mesh_from_model
from cellier.render.points import construct_pygfx_points_from_model


def construct_pygfx_object(visual_model: BaseVisual, data_manager: DataManager):
    """Construct a PyGFX object from a cellier visual model."""
    if isinstance(visual_model, MeshVisual):
        # mesh
        return construct_pygfx_mesh_from_model(
            mesh_model=visual_model, data_manager=data_manager
        )
    elif isinstance(visual_model, PointsVisual):
        # points
        return construct_pygfx_points_from_model(
            mesh_model=visual_model, data_manager=data_manager
        )
    else:
        raise TypeError(f"Unsupported visual model: {type(visual_model)}")
