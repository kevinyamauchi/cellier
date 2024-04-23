"""Model to express a scene."""

from typing import List
from uuid import uuid4

from psygnal import EventedModel
from pydantic import Field

from cellier.models.scene.canvas import Canvas
from cellier.models.scene.dims_manager import DimsManager
from cellier.models.visuals.mesh_visual import MeshVisual


class Scene(EventedModel):
    """Model to express a single scene.

    A scene has a set of visuals and has a single coordinate
    system. A single scene may have multiple canvases that provide
    different views onto that scene.
    """

    dims: DimsManager
    visuals: List[MeshVisual]
    canvases: List[Canvas]

    # store a UUID to identify this specific scene.
    id: str = Field(default_factory=lambda: uuid4().hex)
