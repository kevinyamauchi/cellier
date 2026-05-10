"""Visuals for representing image data."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cellier.v2.transform import AffineTransform
from cellier.v2.visuals._base_visual import BaseVisual
from cellier.v2.visuals._channel_appearance import ChannelAppearance
from cellier.v2.visuals._image_memory import BaseImageAppearance


class MultiscaleImageAppearance(BaseImageAppearance):
    """Appearance parameters for a multiscale image visual.

    Extends ``BaseImageAppearance`` with LOD-control and volume rendering
    fields needed for multiscale brick-streamed visuals.

    Parameters
    ----------
    color_map : cmap.Colormap
        Inherited from ``BaseImageAppearance``.
    clim : tuple[float, float]
        Inherited from ``BaseImageAppearance``.
    interpolation : str
        Inherited from ``BaseImageAppearance``. Controls the shader sampler filter.
    lod_bias : float
        Multiplier on the screen-space LOD threshold. Default 1.0.
    force_level : int | None
        Overrides automatic LOD selection when set. Default None.
    frustum_cull : bool
        Skip bricks outside the camera frustum. Default True.
    iso_threshold : float
        Isosurface threshold for 3D raycast rendering. Default 0.2.
    render_mode : str
        Volume rendering mode. Default ``"iso"``.
    attenuation : float
        Depth attenuation coefficient for ``"attenuated_mip"`` mode.
        Default 1.0.
    """

    lod_bias: float = 1.0
    force_level: int | None = None
    frustum_cull: bool = True
    iso_threshold: float = 0.2
    render_mode: Literal["iso", "mip", "smooth_iso", "attenuated_mip"] = "iso"
    attenuation: float = 1.0


class MultiscaleImageRenderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    """Render-layer configuration for a multiscale image visual.

    These parameters control GPU resource allocation and shader selection.
    They are stored in the model so they round-trip through serialization.

    Parameters
    ----------
    block_size : int
        Brick / tile side length in voxels. Default 32.
    gpu_budget_bytes : int
        Maximum GPU memory for the 3-D brick cache. Default 1 GiB.
    gpu_budget_bytes_2d : int
        Maximum GPU memory for the 2-D tile cache. Default 64 MiB.
    paint_max_tiles : int
        Maximum number of finest-level tiles that can be simultaneously
        painted on during one paint session.  Each slot consumes
        ``block_size**2 * 2 * 4`` bytes of GPU memory (e.g. 8 KB at
        block_size=32).  Default 512 = 4 MB.  When exhausted during a
        session, further paint is staged to the WriteBuffer (and persisted
        on commit) but invisible until commit.
    """

    block_size: int = 32
    gpu_budget_bytes: int = 1 * 1024**3
    gpu_budget_bytes_2d: int = 64 * 1024**2
    paint_max_tiles: int = 512


class MultiscaleImageVisual(BaseVisual):
    """Model for a multiscale image visual.

    Parameters
    ----------
    visual_type : Literal["multiscale_image"]
        Discriminator field. Always ``"multiscale_image"``.
    name : str
        The name of the visual
    data_store_id : str
        The id of the data store to be visualized.
    level_transforms : list[AffineTransform]
        Per-level transforms mapping level-k voxel coords to level-0
        voxel coords.  ``level_transforms[0]`` is the identity.
    appearance : MultiscaleImageAppearance
        The material to use for the labels visual.
    pick_write : bool
        If True, the visual can be picked.
        Default value is True.
    id : str
        The unique id of the visual.
        The default value is a uuid4-generated hex string.
        Do not populate this field manually.
    """

    visual_type: Literal["multiscale_image"] = "multiscale_image"
    level_transforms: list[AffineTransform]
    appearance: MultiscaleImageAppearance
    render_config: MultiscaleImageRenderConfig = Field(
        default_factory=MultiscaleImageRenderConfig
    )
    requires_camera_reslice: bool = Field(default=True, frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _migrate_downscale_factors(cls, data: Any) -> Any:
        """Convert legacy ``downscale_factors`` to ``level_transforms``."""
        if (
            isinstance(data, dict)
            and "downscale_factors" in data
            and "level_transforms" not in data
        ):
            factors = data.pop("downscale_factors")
            ndim = 3  # historical default
            transforms: list[AffineTransform] = []
            for k, f in enumerate(factors):
                s = float(f)
                if k == 0:
                    transforms.append(AffineTransform.identity(ndim=ndim))
                else:
                    scale = tuple(s for _ in range(ndim))
                    translation = tuple((s - 1) / 2 for _ in range(ndim))
                    transforms.append(
                        AffineTransform.from_scale_and_translation(
                            scale=scale, translation=translation
                        )
                    )
            data["level_transforms"] = transforms
        return data


class MultichannelMultiscaleImageVisual(BaseVisual):
    """Model-layer visual for a multichannel multiscale image.

    Parameters
    ----------
    channel_axis : int
        Data-axis index for the channel dimension. Immutable after construction.
    channels : dict[int, ChannelAppearance]
        Per-channel appearance; keys are channel indices.
    interpolation : str
        Texture sampler filter applied to all channels. ``"nearest"`` or
        ``"linear"``. Default ``"nearest"``.
    level_transforms : list[AffineTransform]
        Per-level voxel-level-k → voxel-level-0 AffineTransforms.
    render_config : MultiscaleImageRenderConfig
        Render-layer brick cache and LOD configuration.
    max_channels_2d : int
        2D node pool size. Default 8.
    max_channels_3d : int
        3D node pool size. Default 4.
    requires_camera_reslice : bool
        Always ``True`` (frozen). Camera movement triggers reslicing.
    """

    visual_type: Literal["multichannel_multiscale_image"] = (
        "multichannel_multiscale_image"
    )
    channel_axis: int = Field(frozen=True)
    channels: dict[int, ChannelAppearance]
    interpolation: Literal["linear", "nearest"] = "nearest"
    level_transforms: list[AffineTransform]
    render_config: MultiscaleImageRenderConfig = Field(
        default_factory=MultiscaleImageRenderConfig
    )
    max_channels_2d: int = 8
    max_channels_3d: int = 4
    requires_camera_reslice: bool = Field(default=True, frozen=True)
