"""Model to express a scene."""

from typing import List

from psygnal import EventedModel

from cellier.models.scene.canvas import Canvas
from cellier.models.scene.dims_manager import DimsManager
from cellier.models.visuals.base_visual import BaseVisual


class Scene(EventedModel):
    """Model to express a single scene.

    A scene has a set of visuals and has a single coordinate
    system. A single scene may have multiple canvases that provide
    different views onto that scene.
    """

    dims: DimsManager
    visuals: List[BaseVisual]
    canvases: List[Canvas]
