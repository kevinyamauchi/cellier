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
    id : UUID4
        The unique identifier for the visual.
        The default value is a UUID4 id.
    appearance : BaseAppearance
        The appearance of the visual.
        This should be overridden with the visual-specific
        implementation in the subclasses.
    pick_write : bool
        If True, the visual can be picked in the canvas via
        the picking buffer.
        Default value is True.
    name : str
        The name of the data store.
    transform : AffineTransform
        The data-to-world affine transform.
        Default is identity.

    Attributes
    ----------
    id : str
        The unique identifier for the data store.
    """

    name: str
    appearance: BaseAppearance
    pick_write: bool = True
    transform: AffineTransform = Field(default_factory=AffineTransform.identity)

    requires_camera_reslice: bool = Field(default=False, frozen=True)

    # store a UUID to identify this specific visual
    id: UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))] = (
        Field(frozen=True, default_factory=lambda: uuid4())
    )
