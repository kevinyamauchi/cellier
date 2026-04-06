"""Rendering performance configuration models."""

from __future__ import annotations

from pydantic import BaseModel, Field

DEFAULT_CAMERA_SETTLE_THRESHOLD_S: float = 0.3


class SlicingConfig(BaseModel):
    """Configuration for the async chunk-slicing pipeline.

    These parameters are construction-time only. Changing them after
    ``RenderManager`` is created has no effect.

    Parameters
    ----------
    batch_size : int
        Number of chunks fetched concurrently in each async batch.
        Higher values increase throughput but raise peak memory pressure.
    render_every : int
        Number of completed batches between progressive redraws.
        1 = redraw after every batch (lowest latency to first pixels);
        higher values reduce GPU upload overhead on fast I/O.
    """

    batch_size: int = Field(default=8, gt=0)
    render_every: int = Field(default=1, gt=0)


class TemporalAccumulationConfig(BaseModel):
    """Configuration for the temporal accumulation post-processing pass.

    Parameters
    ----------
    enabled : bool
        When ``False`` the pass is bypassed entirely and each frame is
        shown raw. Useful for debugging or when jitter is disabled.
    alpha : float
        Minimum EMA blend weight for the current frame. During warm-up
        the weight is ``1 / (frame_count + 1)``; once that falls below
        ``alpha`` the weight clamps to ``alpha``. Lower values give
        smoother steady-state but slower convergence after a camera
        move. Must be in ``(0, 1]``.
    """

    enabled: bool = True
    alpha: float = Field(default=0.1, gt=0.0, le=1.0)


class CameraConfig(BaseModel):
    """Configuration for camera-driven automatic reslicing.

    Parameters
    ----------
    reslice_enabled : bool
        When ``False`` camera movement never triggers a reslice.
        Manual calls to ``CellierController.reslice_scene`` still work.
    settle_threshold_s : float
        Seconds of camera stillness required before a reslice is
        triggered. Lower values give more responsive LOD updates;
        higher values reduce redundant I/O during fast panning.
    """

    reslice_enabled: bool = True
    settle_threshold_s: float = Field(default=DEFAULT_CAMERA_SETTLE_THRESHOLD_S, gt=0.0)


class RenderManagerConfig(BaseModel):
    """Top-level rendering performance configuration.

    Pass an instance to ``CellierController(render_config=...)`` at
    construction time. The live state is always accessible and
    serializable via ``render_manager.config``.

    Parameters
    ----------
    slicing : SlicingConfig
        Async chunk-slicing pipeline settings.
    temporal : TemporalAccumulationConfig
        Temporal accumulation pass settings.
    camera : CameraConfig
        Camera-driven reslicing settings.

    Examples
    --------
    Construct with custom settings and serialize:

    >>> config = RenderManagerConfig(
    ...     slicing=SlicingConfig(batch_size=32, render_every=4),
    ...     temporal=TemporalAccumulationConfig(alpha=0.05),
    ...     camera=CameraConfig(settle_threshold_s=0.5),
    ... )
    >>> json_str = config.model_dump_json()
    >>> config2 = RenderManagerConfig.model_validate_json(json_str)
    """

    slicing: SlicingConfig = Field(default_factory=SlicingConfig)
    temporal: TemporalAccumulationConfig = Field(
        default_factory=TemporalAccumulationConfig
    )
    camera: CameraConfig = Field(default_factory=CameraConfig)
