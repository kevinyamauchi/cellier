"""PyGfx shaders for rendering labels images/volumes."""

import pygfx as gfx
from pygfx.renderers.wgpu import register_wgpu_render_function
from pygfx.renderers.wgpu.shaders.imageshader import ImageShader
from pygfx.renderers.wgpu.shaders.volumeshader import VolumeRayShader

from cellier.render.shaders.wgsl import load_wgsl


class LabelImageMaterial(gfx.ImageBasicMaterial):
    """A PyGfx material for rendering label images.

    Parameters
    ----------
    color_map : gfx.TextureMap
        The texture map to use for the coloring the labels.
        This must be a 1D texture and should use nearest neighbor
        interpolation.
    """

    def __init__(self, color_map: gfx.TextureMap):
        # the texture must be 1D
        if color_map.texture.dim != 1:
            raise ValueError("The color map must be a 1D texture.")

        # get the number of colors in the colormap
        n_colors = color_map.texture.size[0]

        # set the clim based on the number of colors
        clim = (0, n_colors - 1)
        super().__init__(
            clim=clim, map=color_map, interpolation="nearest", pick_write=True
        )


class LabelIsoMaterial(gfx.VolumeIsoMaterial):
    """A PyGfx material for rendering label volumes.

    Parameters
    ----------
    color_map : gfx.TextureMap
        The texture map to use for the coloring the labels.
        This must be a 1D texture and should use nearest neighbor
        interpolation.
    """

    render_mode = "iso_categorical"

    def __init__(self, color_map: gfx.TextureMap):
        # the texture must be 1D
        if color_map.texture.dim != 1:
            raise ValueError("The color map must be a 1D texture.")

        # get the number of colors in the colormap
        n_colors = color_map.texture.size[0]

        # set the clim based on the number of colors
        clim = (0, n_colors - 1)

        super().__init__(
            clim=clim,
            map=color_map,
            pick_write=True,
            shininess=0.0,
            interpolation="nearest",
        )


@register_wgpu_render_function(gfx.Image, LabelImageMaterial)
class LabelImageShader(ImageShader):
    """PyGfx shader class for 2D label images."""

    def get_code(self):
        """Load the shader code."""
        return load_wgsl("label_image.wgsl")


@register_wgpu_render_function(gfx.Volume, LabelIsoMaterial)
class LabelIsoShader(VolumeRayShader):
    """PyGfx shader class for 3D label images."""

    def get_code(self):
        """Load the shader code."""
        return load_wgsl("volume_ray.wgsl")
