from __future__ import annotations

from typing import Literal

from pydantic import Field

from cellier.v2.visuals._base_visual import BaseAppearance, BaseVisual


class BaseLabelsAppearance(BaseAppearance):
    """Base appearance parameters shared by all label visuals.

    Parameters
    ----------
    colormap_mode : "random" or "direct"
        Frozen — raises ``ValidationError`` on mutation.
    background_label : int
        Label ID treated as transparent (discarded). Default 0.
    salt : int
        Hash seed for random colormap mode. Default 0.
    color_dict : dict
        Explicit label-ID → RGBA mapping for direct mode.
    render_mode : "iso_categorical" or "flat_categorical"
        3D rendering mode.
    """

    colormap_mode: Literal["random", "direct"] = Field(default="random", frozen=True)
    background_label: int = 0
    salt: int = 0
    color_dict: dict[int, tuple[float, float, float, float]] = Field(
        default_factory=dict
    )
    render_mode: Literal["iso_categorical", "flat_categorical"] = "iso_categorical"


class InMemoryLabelsAppearance(BaseLabelsAppearance):
    """Appearance parameters for an in-memory label visual."""


class LabelMemoryVisual(BaseVisual):
    """Model-layer visual for in-memory label arrays.

    Parameters
    ----------
    appearance : InMemoryLabelsAppearance
        Colormap and rendering appearance.
    """

    visual_type: Literal["label_memory"] = "label_memory"
    appearance: InMemoryLabelsAppearance
    requires_camera_reslice: bool = Field(default=False, frozen=True)
