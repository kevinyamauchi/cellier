from __future__ import annotations

from typing import Literal

from pydantic import Field

from cellier.v2.visuals._base_visual import AABBParams, BaseAppearance, BaseVisual


class LabelMemoryAppearance(BaseAppearance):
    """Appearance for in-memory label visuals.

    Parameters
    ----------
    colormap_mode : "random" or "direct"
        Frozen after construction — changing requires a new visual.
    background_label : int
        Label ID treated as transparent (discarded). Default 0.
    salt : int
        Hash seed for random colormap mode. Default 0.
    color_dict : dict
        Explicit label-ID → RGBA mapping for direct mode.
    render_mode : "iso_categorical" or "flat_categorical"
        3D rendering mode.
    """

    colormap_mode: Literal["random", "direct"] = "random"
    background_label: int = 0
    salt: int = 0
    color_dict: dict[int, tuple[float, float, float, float]] = Field(
        default_factory=dict
    )
    render_mode: Literal["iso_categorical", "flat_categorical"] = "iso_categorical"


class LabelMemoryVisual(BaseVisual):
    """Model-layer visual for in-memory label arrays.

    Parameters
    ----------
    data_store_id : str
        UUID string of a registered LabelMemoryStore.
    appearance : LabelMemoryAppearance
        Colormap and rendering appearance.
    aabb : AABBParams
        Bounding box wireframe parameters.
    """

    visual_type: Literal["label_memory"] = "label_memory"
    data_store_id: str
    appearance: LabelMemoryAppearance
    aabb: AABBParams = Field(default_factory=AABBParams)
    requires_camera_reslice: bool = Field(default=False, frozen=True)
