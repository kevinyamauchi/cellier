"""Discriminated union of all visual model types for cellier v2."""

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from cellier.v2.visuals._image import MultiscaleImageVisual
from cellier.v2.visuals._image_memory import ImageVisual

VisualType = Annotated[
    Union[MultiscaleImageVisual, ImageVisual],
    Field(discriminator="visual_type"),
]
