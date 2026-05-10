import uuid
from typing import Annotated, Literal
from uuid import uuid4

from psygnal import EventedModel
from pydantic import UUID4, AfterValidator, Field

from cellier.v2.transform import AffineTransform


class AABBParams(EventedModel):
    """Parameters for an axis-aligned bounding box wireframe overlay.

    Parameters
    ----------
    enabled : bool
        If True, display the AABB wireframe. Default False.
    color : str
        Line color as a CSS color string. Default ``"#ffffff"``.
    line_width : float
        Line thickness in screen pixels. Default ``2.0``.
    """

    enabled: bool = False
    color: str = "#ffffff"
    line_width: float = 2.0


class BaseAppearance(EventedModel):
    """Base model for all materials.

    Parameters
    ----------
    visible : bool
        If True, the visual is visible.
        Default value is True.
    opacity : float
        Master opacity multiplier in [0, 1].  Default 1.0.
    render_order : int
        Pygfx node render order.  Objects with higher values are drawn later
        and therefore appear on top when depth values are equal.  Default 0.
    """

    visible: bool = True
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    render_order: int = 0
    depth_test: bool = True
    depth_write: bool = True
    depth_compare: Literal["<", "<=", "==", "!=", ">=", ">"] = "<"
    transparency_mode: Literal["blend", "add", "weighted_blend", "weighted_solid"] = (
        "blend"
    )


class BaseVisual(EventedModel):
    """The base class for all Visuals.

    Parameters
    ----------
    name : str
        Human-readable label for the visual.
    data_store_id : str
        UUID string of the data store this visual reads from.
    pick_write : bool
        If True, the visual can be picked in the canvas via the picking
        buffer. Default True.
    transform : AffineTransform
        The data-to-world affine transform. Default is identity.
    aabb : AABBParams
        Axis-aligned bounding box wireframe parameters. Default disabled.
    id : UUID4
        Unique identifier for the visual. Auto-generated; do not set manually.

    Notes
    -----
    Each concrete visual subclass declares its own typed ``appearance`` field.
    Multichannel visuals do not carry a single ``appearance`` field; their
    per-channel appearance is held in ``channels: dict[int, ChannelAppearance]``.
    """

    name: str
    data_store_id: str
    pick_write: bool = True
    transform: AffineTransform = Field(default_factory=AffineTransform.identity)
    requires_camera_reslice: bool = Field(default=False, frozen=True)
    aabb: AABBParams = Field(default_factory=AABBParams)

    # store a UUID to identify this specific visual
    id: UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))] = (
        Field(frozen=True, default_factory=lambda: uuid4())
    )
