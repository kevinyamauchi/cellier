"""Visual for display images."""

from typing import Literal

from cmap import Colormap

from cellier.models.visuals.base import BaseAppearance, BaseVisual


class ImageAppearance(BaseAppearance):
    """Material for a image visual.

    Parameters
    ----------
    color_map : cmap.Colormap
        The color map to use for the labels. This is a cmap Colormap object.
        You can pass the object or the name of a cmap colormap as a string.
        https://cmap-docs.readthedocs.io/en/stable/
    visible : bool
        If True, the visual is visible.
        Default value is True.
    """

    color_map: Colormap


class MultiscaleImageVisual(BaseVisual):
    """Model for a multiscale image visual.

    Parameters
    ----------
    name : str
        The name of the visual
    data_store_id : str
        The id of the data store to be visualized.
    downscale_factors : list[int]
        The downscale factors for each scale level of the labels.
    appearance : ImageAppearance
        The material to use for the labels visual.
    pick_write : bool
        If True, the visual can be picked.
        Default value is True.
    id : str
        The unique id of the visual.
        The default value is a uuid4-generated hex string.
        Do not populate this field manually.
    """

    data_store_id: str
    downscale_factors: list[int]
    appearance: ImageAppearance

    # this is used for a discriminated union
    visual_type: Literal["image"] = "image"
