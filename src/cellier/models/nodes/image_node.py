"""Model for the Image node and image materials."""

from typing import Literal, Tuple

from cellier.models.nodes.base_node import BaseMaterial, BaseNode


class ImageMIPMaterial(BaseMaterial):
    """Render the image with the maximum intensity projection.

    Parameters
    ----------
    clim : Tuple[float, float]
        The contrast limits. The colormap is scaled between
        the (lower_bound, upper_bound).
    """

    clim: Tuple[float, float] = (0, 1)


class ImageNode(BaseNode):
    """Model for an image visual.

    This is a psygnal EventedModel.
    https://psygnal.readthedocs.io/en/latest/API/model/

    Parameters
    ----------
    name : str
        The name of the visual
    data_stream_id : str
        The id of the data stream to be visualized.
    material : ImageMIPMaterial
        The model for the appearance of the rendered image.
    id : str
        The unique id of the visual.
        The default value is a uuid4-generated hex string.
    """

    data_stream_id: str
    material: ImageMIPMaterial

    # this is used for a discriminated union
    visual_type: Literal["image"] = "image"
