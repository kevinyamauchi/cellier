"""Discriminated union of all visual model types for cellier v2."""

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from cellier.v2.visuals._image import (
    MultichannelMultiscaleImageVisual,
    MultiscaleImageVisual,
)
from cellier.v2.visuals._image_memory import ImageVisual, MultichannelImageVisual
from cellier.v2.visuals._lines_memory import LinesVisual
from cellier.v2.visuals._mesh_memory import MeshVisual
from cellier.v2.visuals._points_memory import PointsVisual

VisualType = Annotated[
    Union[
        MultiscaleImageVisual,
        ImageVisual,
        MultichannelImageVisual,
        MultichannelMultiscaleImageVisual,
        PointsVisual,
        LinesVisual,
        MeshVisual,
    ],
    Field(discriminator="visual_type"),
]
