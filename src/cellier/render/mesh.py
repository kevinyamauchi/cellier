"""Functions to make PyGFX mesh objects from Cellier models."""

import pygfx as gfx
from pygfx.materials import MeshPhongMaterial as GFXMeshPhongMaterial

from cellier.models.data_manager import DataManager
from cellier.models.visuals.mesh_visual import MeshPhongMaterial, MeshVisual


def construct_pygfx_mesh_from_model(
    mesh_model: MeshVisual, data_manager: DataManager
) -> gfx.WorldObject:
    """Make a PyGFX mesh object.

    This function dispatches to other constructor functions
    based on the material, etc. and returns a PyGFX mesh object.
    """
    # make the geometry
    # todo make initial slicing happen here
    data_stream = data_manager.streams[mesh_model.data_stream_id]
    data_store = data_manager.stores[data_stream.data_store_id]
    geometry = gfx.Geometry(indices=data_store.faces, positions=data_store.vertices)

    # make the material model
    material_model = mesh_model.material
    if isinstance(material_model, MeshPhongMaterial):
        material = GFXMeshPhongMaterial(
            shininess=material_model.shininess,
            specular=material_model.specular,
            emissive=material_model.emissive,
        )
    else:
        raise TypeError(
            f"Unknown mesh material model type: {type(material_model)} in {mesh_model}"
        )
    return gfx.Mesh(geometry=geometry, material=material)
