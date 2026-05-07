"""Visuals for representing multiscale label data."""

from typing import Literal

from psygnal import EventedModel
from pydantic import Field

from cellier.v2.transform import AffineTransform
from cellier.v2.visuals._base_visual import AABBParams, BaseAppearance, BaseVisual


class LabelAppearance(BaseAppearance):
    """Appearance for multiscale label visuals.

    Extends LabelMemoryAppearance with LOD-control fields.

    Parameters
    ----------
    colormap_mode : "random" | "direct"
        How to assign colors to label IDs.  Frozen after construction.
    background_label : int
        Label ID treated as transparent / background. Default 0.
    salt : int
        Hash seed for random-mode colormap. Default 0.
    color_dict : dict
        Explicit label-ID → RGBA mapping (direct mode only).
    render_mode : "iso_categorical" | "flat_categorical"
        3D rendering mode.
    lod_bias : float
        Multiplier on the screen-space LOD threshold. Default 1.0.
    force_level : int | None
        When set, overrides automatic LOD selection. Default None.
    frustum_cull : bool
        When True, bricks outside the camera frustum are skipped. Default True.
    """

    colormap_mode: Literal["random", "direct"] = "random"
    background_label: int = 0
    salt: int = 0
    color_dict: dict[int, tuple[float, float, float, float]] = Field(
        default_factory=dict
    )
    render_mode: Literal["iso_categorical", "flat_categorical"] = "iso_categorical"
    lod_bias: float = 1.0
    force_level: int | None = None
    frustum_cull: bool = True


class MultiscaleLabelRenderConfig(EventedModel):
    """Render-layer configuration for a multiscale label visual.

    Parameters
    ----------
    block_size : int
        Brick / tile side length in voxels. Default 32.
    gpu_budget_bytes : int
        Maximum GPU memory for the 3-D brick cache. Default 1 GiB.
    gpu_budget_bytes_2d : int
        Maximum GPU memory for the 2-D tile cache. Default 64 MiB.
    """

    block_size: int = 32
    gpu_budget_bytes: int = 1 * 1024**3
    gpu_budget_bytes_2d: int = 64 * 1024**2


class MultiscaleLabelVisual(BaseVisual):
    """Model for a multiscale label visual.

    Parameters
    ----------
    visual_type : Literal["multiscale_label"]
        Discriminator field. Always ``"multiscale_label"``.
    data_store_id : str
        The id of the OMEZarrLabelDataStore to visualize.
    level_transforms : list[AffineTransform]
        Per-level transforms mapping level-k voxel coords to level-0.
    appearance : LabelAppearance
        Visual appearance configuration.
    aabb : AABBParams
        Bounding-box wireframe parameters.
    render_config : MultiscaleLabelRenderConfig
        GPU resource configuration.
    requires_camera_reslice : bool
        Always True (camera movement triggers reslice for LOD). Frozen.
    """

    visual_type: Literal["multiscale_label"] = "multiscale_label"
    data_store_id: str
    level_transforms: list[AffineTransform]
    appearance: LabelAppearance
    aabb: AABBParams = Field(default_factory=AABBParams)
    render_config: MultiscaleLabelRenderConfig = Field(
        default_factory=MultiscaleLabelRenderConfig
    )
    requires_camera_reslice: bool = Field(default=True, frozen=True)
