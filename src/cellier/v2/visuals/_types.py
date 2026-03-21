"""Discriminated union of all visual model types for cellier v2."""

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from cellier.v2.visuals._image import MultiscaleImageVisual

VisualType = Annotated[
    Union[MultiscaleImageVisual,],  # extend as new visual types are added
    Field(discriminator="visual_type"),
]
