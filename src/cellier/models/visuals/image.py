"""Visual for display images."""

import uuid
from typing import Annotated, Literal

from cmap import Colormap
from pydantic import UUID4, AfterValidator, Field

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
    id : UUID4
        The unique identifier for the visual.
        The default value is a UUID4 id.
    appearance : ImageAppearance
        The appearance of the visual.
        This should be overridden with the visual-specific
        implementation in the subclasses.
    pick_write : bool
        If True, the visual can be picked in the canvas via
        the picking buffer.
        Default value is True.
    name : str
        The name of the visual.
    data_store_id : UUID
        The unique identifier for the data store that this visual is
        rendering from.
    """

    visual_type: Literal["multiscale_image"] = "multiscale_image"
    data_store_id: (
        UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))]
    ) = Field(default_factory=lambda: uuid.uuid4())
    appearance: ImageAppearance
