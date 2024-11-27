"""PyGFX implementation of the Image node."""

from typing import Tuple

import numpy as np
import pygfx as gfx
from pygfx.materials import VolumeMinipMaterial as GFXMIPMaterial
from pygfx.materials import VolumeRayMaterial as GFXVolumeRayMaterial

from cellier.models.nodes.image_node import ImageMIPMaterial, ImageNode
from cellier.slicer.data_slice import RenderedImageDataSlice


def construct_pygfx_image_from_model(
    model: ImageNode,
    empty_material: GFXVolumeRayMaterial,
) -> Tuple[gfx.WorldObject, gfx.VolumeRayMaterial]:
    """Make a PyGFX image object.

    This function dispatches to other constructor functions
    based on the material, etc. and returns a PyGFX mesh object.
    """
    # make the geometry
    # todo make initial slicing happen here or initialize with something more sensible

    # initialize with a dummy image
    # since we can't initialize an empty node.
    geometry = gfx.Geometry(
        grid=np.ones((5, 5, 5), dtype=np.float32),
    )

    # make the material model
    material_model = model.material
    if isinstance(material_model, ImageMIPMaterial):
        material = GFXMIPMaterial(clim=material_model.clim, map=gfx.cm.magma)
    else:
        raise TypeError(
            f"Unknown mesh material model type: {type(material_model)} in {model}"
        )
    return gfx.Volume(geometry=geometry, material=empty_material), material


class GFXImageNode:
    """PyGFX image node implementation.

    Note that PyGFX doesn't support empty WorldObjects, so we set
    transparent data when the slice is empty.
    """

    def __init__(self, model: ImageNode):
        # This is the material given when the visual is "empty"
        # since pygfx doesn't support empty World Objects, we
        # initialize with a single point
        self._empty_material = GFXVolumeRayMaterial()

        # make the pygfx materials
        self.node, self._material = construct_pygfx_image_from_model(
            model=model, empty_material=self._empty_material
        )

        # Flag that is set to True when there are no points to display.
        self._empty = True

    @property
    def material(self) -> GFXVolumeRayMaterial:
        """The material object points."""
        return self._material

    def set_slice(self, slice_data: RenderedImageDataSlice):
        """Set all the point coordinates."""
        data = slice_data.data

        # check if the layer was empty
        was_empty = self._empty
        if data.ndim == 2:
            # account for the 2D case
            # todo: add 2D rendering...
            data = np.atleast_3d(data)

        if data.size == 0:
            # coordinates must not be empty
            # todo do something smarter?
            data = np.ones((5, 5, 5), dtype=np.float32)

            # set the empty flag
            self._empty = True
        else:
            # There is data to set, so the node is not empty
            self._empty = False

        new_geometry = gfx.Geometry(grid=data)
        self.node.geometry = new_geometry

        if was_empty and not self._empty:
            # if this is the first data after the layer
            # was empty, set the material
            self.node.material = self.material
        elif not was_empty and self._empty:
            # if the layer has become empty, set the material
            self.node.material = self._empty_material
