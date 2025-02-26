"""Models for nodes."""

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from cellier.models.visuals.lines import LinesUniformMaterial, LinesVisual
from cellier.models.visuals.points import PointsUniformMaterial, PointsVisual

VisualType = Annotated[
    Union[LinesVisual, PointsVisual],
    Field(discriminator="visual_type"),
]

__all__ = [
    "PointsVisual",
    "LinesVisual",
    "VisualType",
    "PointsUniformMaterial",
    "LinesUniformMaterial",
]
