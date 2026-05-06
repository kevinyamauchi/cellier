# src/cellier/v2/visuals/_image_memory.py
from typing import Literal

from cmap import Colormap
from pydantic import Field, field_validator

from cellier.v2.visuals._base_visual import AABBParams, BaseAppearance, BaseVisual
from cellier.v2.visuals._channel_appearance import ChannelAppearance

__all__ = [
    "ChannelAppearance",
    "ImageMemoryAppearance",
    "ImageVisual",
    "MultichannelImageVisual",
]


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
    interpolation: Literal["linear", "nearest"] = "nearest"


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
    aabb: AABBParams = Field(default_factory=AABBParams)

    requires_camera_reslice: bool = Field(default=False, frozen=True)


class MultichannelImageVisual(BaseVisual):
    """Model-layer visual for a multichannel in-memory image.

    Parameters
    ----------
    channel_axis : int
        Data-axis index that corresponds to the channel dimension.
        Immutable after construction.
    channels : dict[int, ChannelAppearance]
        Maps each channel index to its appearance.
    data_store_id : str
        UUID (as a string) of the ``ImageMemoryStore`` this visual reads from.
    max_channels_2d : int
        Size of the 2D node pool. Default 8.
    max_channels_3d : int
        Size of the 3D node pool. Default 4.
    requires_camera_reslice : bool
        Always ``False`` (frozen).
    """

    visual_type: Literal["multichannel_image_memory"] = "multichannel_image_memory"
    channel_axis: int = Field(frozen=True)
    channels: dict[int, ChannelAppearance]
    data_store_id: str
    aabb: AABBParams = Field(default_factory=AABBParams)
    max_channels_2d: int = 8
    max_channels_3d: int = 4
    requires_camera_reslice: bool = Field(default=False, frozen=True)

    # BaseVisual requires an `appearance` field; provide a minimal placeholder.
    appearance: BaseAppearance = Field(default_factory=BaseAppearance)

    @field_validator("channels")
    @classmethod
    def _validate_channel_count(cls, v: dict) -> dict:
        return v
