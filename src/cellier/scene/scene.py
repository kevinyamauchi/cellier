"""Scene model for cellier v2."""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal
from uuid import uuid4

from psygnal import EventedModel
from pydantic import UUID4, AfterValidator, Field, PrivateAttr, field_serializer

from cellier.scene._background import BackgroundAppearance
from cellier.scene.canvas import Canvas
from cellier.scene.dims import DimsManager
from cellier.visuals._types import VisualType


class Scene(EventedModel):
    """One rendered scene.

    Parameters
    ----------
    id : UUID4
        Unique identifier. Auto-generated.
    name : str
        Human-readable name, e.g. ``"main"``.
    dims : DimsManager
        Dimension manager; single source of truth for render dimensionality.
    visuals : list[VisualType]
        Discriminated union of visual model types.
    canvases : dict[UUID4, Canvas]
        Keyed by ``canvas.id``.
    render_modes : set[Literal["2d", "3d"]]
        Which rendering modes visuals added to this scene should support.
        Defaults to ``{"2d", "3d"}``.
    lighting : Literal["none", "default"]
        ``"none"`` (default) or ``"default"``.  Pass ``"default"`` to add
        ambient and directional lights — required for MeshPhongAppearance.
    background : BackgroundAppearance
        Appearance of the background drawn behind this scene's visuals.
        Mutating its fields updates the render layer at runtime.
    """

    id: UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))] = (
        Field(frozen=True, default_factory=lambda: uuid4())
    )
    name: str
    dims: DimsManager
    visuals: list[VisualType] = Field(default_factory=list)
    canvases: dict[
        UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))],
        Canvas,
    ] = Field(default_factory=dict)
    render_modes: set[Literal["2d", "3d"]] = Field(default_factory=lambda: {"2d", "3d"})
    lighting: Literal["none", "default"] = "none"
    background: BackgroundAppearance = Field(default_factory=BackgroundAppearance)

    # The background model the relay below is currently attached to.  Needed
    # to tell a nested field change (which re-emits events.background) apart
    # from an actual reassignment of the field.
    _background_relay_target: BackgroundAppearance | None = PrivateAttr(default=None)

    @field_serializer("render_modes")
    def _serialize_render_modes(self, value: set) -> list:
        return sorted(value)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # ``self.events`` only exists once EventedModel.__init__ has built the
        # signal group, which happens after model_post_init runs -- so unlike
        # the relays below, this connection cannot live there.
        self.events.background.connect(self._on_background_assigned)

    def model_post_init(self, __context: Any) -> None:
        """Wire dims, background and visual event relays after initialization."""
        self.dims.events.all.connect(self._on_dims_updated)
        self._connect_background_relay()
        for visual in self.visuals:
            visual.events.all.connect(
                lambda info, v=visual: self.events.visuals.emit(self.visuals)
            )

    def _on_dims_updated(self, info: Any) -> None:
        self.events.dims.emit(self.dims)

    def _connect_background_relay(self) -> None:
        """Relay the current background model's field changes to this scene."""
        self._background_relay_target = self.background
        self.background.events.all.connect(self._on_background_updated)

    def _on_background_assigned(self, value: BackgroundAppearance) -> None:
        """Move the relay onto a newly assigned background model.

        ``events.background`` fires both when a nested field changes (via
        ``_on_background_updated`` below) and when the field itself is
        reassigned.  The identity check separates the two: only a genuinely
        different object needs rewiring, and returning early on the first case
        is also what stops the two handlers recursing into each other.
        """
        if value is self._background_relay_target:
            return
        if self._background_relay_target is not None:
            self._background_relay_target.events.all.disconnect(
                self._on_background_updated
            )
        self._connect_background_relay()

    def _on_background_updated(self, info: Any) -> None:
        self.events.background.emit(self.background)
