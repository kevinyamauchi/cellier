"""Model to express a scene."""

from typing import List, Union
from uuid import uuid4

from psygnal import EventedModel
from pydantic import Field
from typing_extensions import Annotated

from cellier.models.scene.canvas import Canvas
from cellier.models.scene.dims_manager import DimsManager
from cellier.models.visuals.mesh_visual import MeshVisual
from cellier.models.visuals.points_visual import PointsVisual

VisualType = Annotated[
    Union[MeshVisual, PointsVisual], Field(discriminator="visual_type")
]


class Scene(EventedModel):
    """Model to express a single scene.

    A scene has a set of visuals and has a single coordinate
    system. A single scene may have multiple canvases that provide
    different views onto that scene.
    """

    dims: DimsManager
    visuals: List[VisualType]
    canvases: List[Canvas]

    # store a UUID to identify this specific scene.
    id: str = Field(default_factory=lambda: uuid4().hex)
