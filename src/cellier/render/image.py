"""PyGFX implementation of the Image node."""

from typing import Tuple

import numpy as np
import pygfx as gfx
import wgpu
from pygfx.materials import VolumeIsoMaterial as GFXIsoMaterial
from pygfx.materials import VolumeMipMaterial as GFXMIPMaterial
from pygfx.materials import VolumeRayMaterial as GFXVolumeRayMaterial

from cellier.models.nodes.image_node import (
    ImageIsoMaterial,
    ImageMIPMaterial,
    ImageNode,
    MultiscaleImageNode,
)
from cellier.slicer.data_slice import RenderedImageDataSlice


def construct_pygfx_image_from_model(
    model: ImageNode,
    empty_material: GFXVolumeRayMaterial,
) -> Tuple[gfx.WorldObject, gfx.VolumeRayMaterial]:
    """Make a PyGFX image object.

    This function dispatches to other constructor functions
    based on the material, etc. and returns a PyGFX image object.
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
    elif isinstance(material_model, ImageIsoMaterial):
        material = GFXIsoMaterial(
            clim=material_model.clim,
            map=gfx.cm.magma,
            threshold=material_model.iso_threshold,
        )
    else:
        raise TypeError(
            f"Unknown mesh material model type: {type(material_model)} in {model}"
        )
    return gfx.Volume(geometry=geometry, material=empty_material), material


def construct_pygfx_multiscale_image_from_model(
    model: MultiscaleImageNode,
    empty_material: GFXVolumeRayMaterial,
) -> Tuple[gfx.WorldObject, gfx.VolumeRayMaterial]:
    """Make a multiscale PyGFX image object.

    This function dispatches to other constructor functions
    based on the material, etc. and returns a PyGFX Group object.

    The node is structured as follows:
        - root group (name is the model ID)
        - low res volume visual. this is used for
        - "multiscales" group. each subnode of the group is a
            volume from a given scale. The names are scale_i
            where i is the index of the scale level going from
            0 (highest resolution).
    """
    # make the geometry
    # todo make initial slicing happen here or initialize with something more sensible

    # we make the
    node = gfx.Group(name=model.id)

    multiscale_group = gfx.Group(name="multiscales")

    for scale_level_index in range(model.n_scales):
        # initialize with a dummy image
        # since we can't initialize an empty node.
        geometry = gfx.Geometry(
            grid=np.ones((5, 5, 5), dtype=np.float32),
        )
        scale_image = gfx.Volume(
            geometry=geometry,
            material=empty_material,
            name=f"scale_{scale_level_index}",
        )
        scale_image.local.scale = (
            2**scale_level_index,
            2**scale_level_index,
            2**scale_level_index,
        )
        multiscale_group.add(scale_image)

    # add the images to the base node
    node.add(multiscale_group)

    # make the material model
    material_model = model.material
    if isinstance(material_model, ImageMIPMaterial):
        material = GFXMIPMaterial(clim=material_model.clim, map=gfx.cm.magma)
    elif isinstance(material_model, ImageIsoMaterial):
        material = GFXIsoMaterial(
            clim=material_model.clim,
            map=gfx.cm.magma,
            threshold=material_model.iso_threshold,
        )
    else:
        raise TypeError(
            f"Unknown mesh material model type: {type(material_model)} in {model}"
        )
    return node, material


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

        # Flag that is set to True when there are no data to display.
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


class GFXMultiScaleImageNode:
    """PyGFX multiscale image node implementation.

    Note that PyGFX doesn't support empty WorldObjects, so we set
    transparent data when the slice is empty.
    """

    def __init__(self, model: MultiscaleImageNode):
        # This is the material given when the visual is "empty"
        # since pygfx doesn't support empty World Objects, we
        # initialize with a single point
        self._empty_material = GFXVolumeRayMaterial()

        # make the pygfx materials
        self.node, self._material = construct_pygfx_multiscale_image_from_model(
            model=model, empty_material=self._empty_material
        )

        # Flag that is set to True when there are no data to display.
        self._empty = True

    @property
    def material(self) -> GFXVolumeRayMaterial:
        """The material object points."""
        return self._material

    def get_node_by_scale(self, scale_index: int) -> gfx.Volume:
        """Get the node for a specific volume."""
        for child in self.node.children:
            if child.name == "multiscales":
                for scale in child.children:
                    if scale.name == f"scale_{scale_index:}":
                        return scale

    def preallocate_data(
        self,
        scale_index: int,
        shape: Tuple[int, int, int],
        chunk_shape: Tuple[int, int, int],
        translation: Tuple[int, int, int],
    ):
        """Preallocate the data for a given scale."""
        texture = gfx.Texture(
            data=None,
            size=shape,
            format="1xf4",
            usage=wgpu.TextureUsage.COPY_DST,
            dim=3,
            force_contiguous=True,
        )
        scale_node = self.get_node_by_scale(scale_index)

        # set the new texture
        scale_node.geometry = gfx.Geometry(grid=texture)
        scale_node.material = self._material

        # set the translation
        scale_node.local.position = translation

    def set_scale_visible(self, scale_index: int):
        """Set a specified scale level as the currently visible level.

        All other scales are set visible = False.
        """
        for child in self.node.children:
            if child.name == "multiscales":
                for scale in child.children:
                    if scale.name == f"scale_{scale_index:}":
                        scale.visible = True
                    else:
                        scale.visible = False
            else:
                # this is the low res node
                child.visible = False

    def set_slice(self, slice_data: RenderedImageDataSlice):
        """Set all the point coordinates."""
        data = slice_data.data

        # check if the layer was empty
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

        # set the data
        scale_node = self.get_node_by_scale(slice_data.resolution_level)
        texture = scale_node.geometry.grid
        texture.send_data(tuple(slice_data.texture_start_index), slice_data.data)
        scale_node.visible = True
