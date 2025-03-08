"""Types used in the Cellier package."""

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from cellier.models.visuals import LinesVisual, MultiscaleLabelsVisual, PointsVisual

# This is used for a discriminated union for typing the visual models
VisualType = Annotated[
    Union[LinesVisual, PointsVisual, MultiscaleLabelsVisual],
    Field(discriminator="visual_type"),
]
