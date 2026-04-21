"""Discriminated union of all canvas overlay model types for cellier v2."""

from __future__ import annotations

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from cellier.v2.visuals._canvas_overlay import CenteredAxes2D

CanvasOverlayType = Annotated[
    Union[CenteredAxes2D,],
    Field(discriminator="overlay_type"),
]
