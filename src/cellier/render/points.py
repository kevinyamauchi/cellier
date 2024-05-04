"""Functions for constructing PyGFX objects from cellier models."""

import numpy as np
import pygfx as gfx
from pygfx import PointsMaterial as GFXPointsMaterial

from cellier.models.data_manager import DataManager
from cellier.models.visuals.points_visual import PointsUniformMaterial, PointsVisual

cellier_to_gfx_coordinate_space = {
    "data": "model",
    "world": "world",
    "screen": "screen",
}


def construct_pygfx_points_from_model(
    mesh_model: PointsVisual, data_manager: DataManager
) -> gfx.WorldObject:
    """Make a PyGFX mesh object.

    This function dispatches to other constructor functions
    based on the material, etc. and returns a PyGFX mesh object.
    """
    # make the geometry
    # todo make initial slicing happen here or initialize with something more sensible

    # initialize with an empty geometry
    geometry = gfx.Geometry(
        positions=np.array(
            [[0, 0, 0], [0, 10, 0], [0, 0, 10], [5, 1, 10]], dtype=np.float32
        ),
    )

    # make the material model
    material_model = mesh_model.material
    if isinstance(material_model, PointsUniformMaterial):
        size_space = cellier_to_gfx_coordinate_space[
            material_model.size_coordinate_space
        ]
        material = GFXPointsMaterial(
            size=material_model.size,
            size_space=size_space,
            color=material_model.color,
            size_mode="uniform",
        )
    else:
        raise TypeError(
            f"Unknown mesh material model type: {type(material_model)} in {mesh_model}"
        )
    return gfx.Points(geometry=geometry, material=material)
