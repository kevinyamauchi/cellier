"""Functions to make PyGFX mesh objects from Cellier models."""

from typing import Tuple

import numpy as np
import pygfx as gfx
from pygfx.materials import MeshAbstractMaterial
from pygfx.materials import MeshPhongMaterial as GFXMeshPhongMaterial
from pygfx.materials import MeshStandardMaterial as GFXMeshStandardMaterial

from cellier.models.nodes.mesh_node import MeshNode, MeshPhongMaterial
from cellier.slicer.data_slice import RenderedMeshDataSlice


def construct_pygfx_mesh_from_model(
    model: MeshNode, empty_material: MeshAbstractMaterial
) -> Tuple[gfx.WorldObject, MeshAbstractMaterial]:
    """Make a PyGFX mesh object.

    This function dispatches to other constructor functions
    based on the material, etc. and returns a PyGFX mesh object.
    """
    # make the geometry
    # todo make initial slicing happen here or initialize with something more sensible

    # initialize with an empty geometry
    geometry = gfx.Geometry(
        indices=np.array([[0, 1, 2]], dtype=np.int32),
        positions=np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
    )

    # make the material model
    material_model = model.material
    if isinstance(material_model, MeshPhongMaterial):
        material = GFXMeshPhongMaterial(
            shininess=material_model.shininess,
            specular=material_model.specular,
            emissive=material_model.emissive,
        )
    else:
        raise TypeError(
            f"Unknown mesh material model type: {type(material_model)} in {model}"
        )
    return gfx.Mesh(geometry=geometry, material=empty_material), material


class GFXMeshNode:
    """PyGFX mesh node implementation.

    Note that PyGFX doesn't support empty WorldObjects, so we set
    transparent data when the slice is empty.
    """

    def __init__(self, model: MeshNode):
        # This is the material given when the visual is "empty"
        # since pygfx doesn't support empty World Objects, we
        # initialize with a single point
        self._empty_material = GFXMeshStandardMaterial(color=(0, 0, 0, 0))

        # make the pygfx materials
        self.node, self._material = construct_pygfx_mesh_from_model(
            model=model, empty_material=self._empty_material
        )

        # Flag that is set to True when there are no points to display.
        self._empty = True

    @property
    def material(self) -> MeshAbstractMaterial:
        """The material object mesh."""
        return self._material

    def set_slice(self, slice_data: RenderedMeshDataSlice):
        """Set all the point coordinates."""
        vertices = slice_data.vertices
        faces = slice_data.faces

        # check if the layer was empty
        was_empty = self._empty
        if vertices.shape[1] == 2:
            # pygfx expects 3D points
            n_points = vertices.shape[0]
            zeros_column = np.zeros((n_points, 1), dtype=np.float32)
            vertices = np.column_stack((vertices, zeros_column))

        if vertices.shape[0] == 0:
            # coordinates must not be empty
            # todo do something smarter?
            vertices = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
            faces = (np.array([[0, 1, 2]], dtype=np.int32),)

            # set the empty flag
            self._empty = True
        else:
            # There is data to set, so the node is not empty
            self._empty = False

        new_geometry = gfx.Geometry(positions=vertices, indices=faces)
        self.node.geometry = new_geometry

        if was_empty and not self._empty:
            # if this is the first data after the layer
            # was empty, set the material
            self.node.material = self.material
        elif not was_empty and self._empty:
            # if the layer has become empty, set the material
            self.node.material = self._empty_material
