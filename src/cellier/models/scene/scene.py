"""Model to express a scene."""

from typing import List, Union
from uuid import uuid4

from psygnal import EventedModel
from pydantic import Field
from typing_extensions import Annotated

from cellier.models.nodes.mesh_visual import MeshNode
from cellier.models.nodes.points_visual import PointsNode
from cellier.models.scene.canvas import Canvas
from cellier.models.scene.dims_manager import DimsManager

VisualType = Annotated[Union[MeshNode, PointsNode], Field(discriminator="visual_type")]


class Scene(EventedModel):
    """Model to express a single scene.

    A scene has a set of nodes and has a single coordinate
    system. A single scene may have multiple canvases that provide
    different views onto that scene.
    """

    dims: DimsManager
    visuals: List[VisualType]
    canvases: List[Canvas]

    # store a UUID to identify this specific scene.
    id: str = Field(default_factory=lambda: uuid4().hex)
