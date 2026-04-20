"""Canvas model for cellier v2."""

from __future__ import annotations

import uuid
from typing import Annotated, Any
from uuid import uuid4

from psygnal import EventedModel
from pydantic import UUID4, AfterValidator, Field

from cellier.v2.scene.cameras import CameraType
from cellier.v2.visuals._overlay_types import CanvasOverlayType


class Canvas(EventedModel):
    """One rendered canvas, holding one or more cameras.

    Parameters
    ----------
    id : UUID4
        Unique identifier. Auto-generated.
    cameras : dict[str, CameraType]
        Mapping of ``"2d"`` and/or ``"3d"`` to camera models.
        At least one entry is required.
    overlays : list[CanvasOverlayType]
        Screen-space overlays attached to this canvas.  Rendered as
        post-passes on top of the main scene.  Default empty list.
    """

    id: UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))] = (
        Field(frozen=True, default_factory=lambda: uuid4())
    )
    cameras: dict[str, CameraType]
    overlays: list[CanvasOverlayType] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Wire camera event relays after model initialization."""
        for camera in self.cameras.values():
            camera.events.all.connect(
                lambda info, cam=camera: self.events.cameras.emit(self.cameras)
            )
