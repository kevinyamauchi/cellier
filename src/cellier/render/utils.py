"""Utilities for interfacing between cellier models and PyGFX objects."""

from cellier.models.visuals.base_visual import BaseVisual
from cellier.models.visuals.mesh_visual import MeshVisual
from cellier.render.mesh import construct_pygfx_mesh_from_model


def construct_pygfx_object(visual_model: BaseVisual):
    """Construct a PyGFX object from a cellier visual model."""
    if isinstance(visual_model, MeshVisual):
        # mesh
        return construct_pygfx_mesh_from_model(mesh_model=visual_model)
    else:
        raise TypeError(f"Unsupported visual model: {type(visual_model)}")
