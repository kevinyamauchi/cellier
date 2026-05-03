"""Visuals for representing image data."""

from typing import Any, Literal

from cmap import Colormap
from psygnal import EventedModel
from pydantic import Field, model_validator

from cellier.v2.transform import AffineTransform
from cellier.v2.visuals._base_visual import AABBParams, BaseAppearance, BaseVisual


class ImageAppearance(BaseAppearance):
    """Appearance parameters for an image visual.

    Parameters
    ----------
    color_map : cmap.Colormap
        The color map to use for the image. This is a cmap Colormap object.
        You can pass the object or the name of a cmap colormap as a string.
        https://cmap-docs.readthedocs.io/en/stable/
    clim : tuple[float, float]
        The contrast limits (min, max) used to normalise the image data
        before colour mapping.  Default is (0.0, 1.0).
    visible : bool
        If True, the visual is visible.  Default is True.
    lod_bias : float
        Multiplier on the screen-space LOD threshold. Higher values force
        finer levels. Default is 1.0.
    force_level : int | None
        When set, overrides automatic LOD selection and always uses this
        scale index. ``None`` means automatic. Default is None.
    frustum_cull : bool
        When True, bricks outside the camera frustum are skipped during
        brick planning. Default is True.
    iso_threshold : float
        Isosurface threshold for 3D raycast rendering. Default is 0.2.
    render_mode : Literal["iso", "mip"]
        Volume rendering mode.  ``"iso"`` for isosurface rendering,
        ``"mip"`` for Maximum Intensity Projection.  Default is ``"iso"``.
    """

    color_map: Colormap
    clim: tuple[float, float] = (0.0, 1.0)
    lod_bias: float = 1.0
    force_level: int | None = None
    frustum_cull: bool = True
    iso_threshold: float = 0.2
    render_mode: Literal["iso", "mip"] = "iso"


class MultiscaleImageRenderConfig(EventedModel):
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
    interpolation : Literal["linear", "nearest"]
        Sampler filter for the block shader. Default ``"linear"``.
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
    interpolation: Literal["linear", "nearest"] = "nearest"
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
    appearance : ImageAppearance
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
    data_store_id: str
    level_transforms: list[AffineTransform]
    appearance: ImageAppearance
    aabb: AABBParams = Field(default_factory=AABBParams)
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
