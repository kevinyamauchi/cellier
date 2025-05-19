"""PyGFx node for rendering multiscale labels."""

from typing import Callable

import numpy as np
import pygfx as gfx
import wgpu
from cmap import Colormap

from cellier.models.visuals import LabelsMaterial, MultiscaleLabelsVisual
from cellier.render.shaders import LabelImageMaterial, LabelIsoMaterial
from cellier.types import ImageDataResponse


def construct_pygfx_labels_from_model(
    model: MultiscaleLabelsVisual,
) -> tuple[gfx.WorldObject, gfx.Material]:
    """Make a PyGFX multiscale labels object.

    This function dispatches to other constructor functions
    based on the material, etc. and returns a PyGFX image object.
    """
    # make the material
    material_model = model.material
    if isinstance(material_model, LabelsMaterial):
        pygfx_cm = Colormap("glasbey:glasbey").to_pygfx(N=200)
        material_2d = LabelImageMaterial(color_map=pygfx_cm)
        material_3d = LabelIsoMaterial(
            color_map=pygfx_cm,
        )
    else:
        raise TypeError(
            f"Unknown mesh material model type: {type(material_model)} in {model}"
        )

    # make the parent node
    node = gfx.Group(name=model.id)

    # add the 2D multiscales node
    multiscale_group_2d = gfx.Group(name="multiscale_2d")
    for scale_level_index, downscale_factor in enumerate(model.downscale_factors):
        # initialize empty texture
        geometry = gfx.Geometry(
            grid=gfx.Texture(np.empty((1, 1), dtype=np.float32), dim=2),
        )
        scale_image = gfx.Image(
            geometry=geometry,
            material=material_2d,
            name=f"scale_{scale_level_index}",
        )
        scale_image.local.scale = (1, downscale_factor, downscale_factor)
        multiscale_group_2d.add(scale_image)
    node.add(multiscale_group_2d)

    # add the 3D multiscales node
    multiscale_group_3d = gfx.Group(name="multiscale_3d")
    for scale_level_index, downscale_factor in enumerate(model.downscale_factors):
        # initialize empty texture
        geometry = gfx.Geometry(
            grid=gfx.Texture(np.empty((1, 1, 1), dtype=np.float32), dim=3),
        )
        scale_image = gfx.Volume(
            geometry=geometry,
            material=material_3d,
            name=f"scale_{scale_level_index}",
        )
        scale_image.local.scale = (downscale_factor, downscale_factor, downscale_factor)
        multiscale_group_3d.add(scale_image)
    node.add(multiscale_group_3d)

    return node, (material_2d, material_3d)


class GFXMultiScaleLabelsNode:
    """A PyGfx node that renders a set of multiscale labels."""

    def __init__(self, model: MultiscaleLabelsVisual):
        self.node, self._material = construct_pygfx_labels_from_model(model=model)

    @property
    def callback_handlers(self) -> list[Callable]:
        """Return the list of callback handlers for all nodes."""
        callback_handlers = []
        for dim_node in self.node.children:
            # iterate over the 2D and 3D nodes
            for node in dim_node.children:
                # iterate over the scales
                callback_handlers.append(node.add_event_handler)

        return callback_handlers

    def preallocate_data(
        self,
        scale_index: int,
        ndim: int,
        shape: tuple[int, int, int],
        chunk_shape: tuple[int, int, int],
        translation: tuple[int, int, int],
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
        scale_node = self.get_node_by_scale(scale_index, ndim=ndim)

        # set the new texture
        scale_node.geometry = gfx.Geometry(grid=texture)
        scale_node.material = self._material

        # set the translation
        scale_node.local.position = translation

    def get_node_by_scale(self, scale_index: int, ndim: int) -> gfx.Volume:
        """Get the node for a specific volume.

        Parameters
        ----------
        scale_index : int
            The index of the scale to be set as visible.
        ndim : int
            The dimensionality of the visual to set.
            Should be 2 for 2D or 3 for 3D.
        """
        parent_node_name = f"multiscale_{ndim}d"
        node_name = f"scale_{scale_index:}"
        for child in self.node.children:
            if child.name == parent_node_name:
                for scale in child.children:
                    if scale.name == node_name:
                        return scale

    def set_scale_visible(self, scale_index: int, ndim: int):
        """Set a specified scale level as the currently visible level.

        All other scales are set visible = False.

        Parameters
        ----------
        scale_index : int
            The index of the scale to be set as visible.
        ndim : int
            The dimensionality of the visual to set.
            Should be 2 for 2D or 3 for 3D.
        """
        parent_node_name = f"multiscale_{ndim}d"
        node_name = f"scale_{scale_index:}"
        for child in self.node.children:
            if child.name == parent_node_name:
                for scale in child.children:
                    if scale.name == node_name:
                        scale.visible = True
                    else:
                        scale.visible = False
            else:
                # this is a different node.
                child.visible = False

    def set_slice(self, slice_data: ImageDataResponse):
        """Set all the point coordinates."""
        data = slice_data.data

        # check if the layer was empty
        if data.size == 0:
            # coordinates must not be empty
            # todo do something smarter?
            data = np.empty((1, 1, 1), dtype=np.float32)

            # set the empty flag
            self._empty = True
        else:
            # There is data to set, so the node is not empty
            self._empty = False

        # set the data
        scale_node = self.get_node_by_scale(slice_data.resolution_level, data.ndim)

        # if scale_node.geometry.grid is None:
        scale_node.geometry = gfx.Geometry(grid=gfx.Texture(data=data, dim=data.ndim))
        # else:
        #     texture = scale_node.geometry.grid
        #     texture.send_data(tuple(slice_data.texture_start_index), slice_data.data)
        self.set_scale_visible(scale_index=slice_data.resolution_level, ndim=data.ndim)
