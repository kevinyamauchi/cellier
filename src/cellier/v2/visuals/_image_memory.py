# src/cellier/v2/visuals/_image_memory.py
from typing import Literal

from cmap import Colormap
from pydantic import Field

from cellier.v2.visuals._base_visual import BaseAppearance, BaseVisual


class ImageMemoryAppearance(BaseAppearance):
    """Appearance parameters for an in-memory image visual.

    Parameters
    ----------
    color_map : cmap.Colormap
        Colourmap applied after contrast normalisation. Accepts any
        cmap-registered name string (e.g. ``"viridis"``, ``"bids:magma"``).
    clim : tuple[float, float]
        Contrast limits ``(min, max)`` used to normalise pixel values
        before colour-mapping. Default ``(0.0, 1.0)``.
    interpolation : str
        Texture sampler filter. ``"linear"`` (default) or ``"nearest"``.
    visible : bool
        Inherited from ``BaseAppearance``. Default ``True``.
    """

    color_map: Colormap
    clim: tuple[float, float] = (0.0, 1.0)
    interpolation: Literal["linear", "nearest"] = "linear"


class ImageVisual(BaseVisual):
    """Model-layer visual for a single-resolution in-memory image.

    Wraps a pygfx ``gfx.Image`` (2D scene) or ``gfx.Volume`` (3D scene).
    The associated data store must be an ``ImageMemoryStore``.

    Camera movement does **not** trigger a reslice because data is not
    view-dependent — the full slice is always loaded.

    Parameters
    ----------
    visual_type : Literal["image_memory"]
        Discriminator field; always ``"image_memory"``.
    name : str
        Human-readable label.
    data_store_id : str
        UUID (as a string) of the ``ImageMemoryStore`` this visual reads from.
    appearance : ImageMemoryAppearance
        Appearance parameters.
    requires_camera_reslice : bool
        Always ``False``; frozen. Camera movement does not trigger reslicing.
    """

    visual_type: Literal["image_memory"] = "image_memory"
    data_store_id: str
    appearance: ImageMemoryAppearance

    requires_camera_reslice: bool = Field(default=False, frozen=True)
