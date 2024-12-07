"""Model to express a scene."""

from typing import Dict, List, Optional, Union
from uuid import uuid4

from psygnal import EventedModel
from pydantic import Field
from typing_extensions import Annotated

from cellier.models.nodes.image_node import ImageNode, MultiscaleImageNode
from cellier.models.nodes.mesh_node import MeshNode
from cellier.models.nodes.points_node import PointsNode
from cellier.models.scene.canvas import Canvas
from cellier.models.scene.dims_manager import DimsManager

VisualType = Annotated[
    Union[ImageNode, MultiscaleImageNode, MeshNode, PointsNode],
    Field(discriminator="visual_type"),
]


class Scene(EventedModel):
    """Model to express a single scene.

    A scene has a set of nodes and has a single coordinate
    system. A single scene may have multiple canvases that provide
    different views onto that scene.
    """

    dims: DimsManager
    visuals: List[VisualType]

    # key is the canvas id
    canvases: Dict[str, Canvas]

    # store a UUID to identify this specific scene.
    id: str = Field(default_factory=lambda: uuid4().hex)

    def get_visual_by_id(self, visual_id: str) -> Optional[VisualType]:
        """Get a visual from the scene by the visual's id.

        Parameters
        ----------
        visual_id : str
            The ID of the visual you want to retrieve.

        Returns
        -------
        Optional[VisualType]
            The requested visual. Returns None if the ID is not found.
        """
        for visual in self.visuals:
            if visual.id == visual_id:
                return visual

        # if the visual isn't found, return None
        return None
