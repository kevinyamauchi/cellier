"""Per-visual render configuration for the RenderManager pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VisualRenderConfig:
    """Mutable render settings for one visual.

    Passed through the call stack at reslice time.  When a visual's ID is
    absent from the ``visual_configs`` dict supplied to
    ``SceneManager.build_slice_requests``, a default instance is used.

    Parameters
    ----------
    lod_bias : float
        Multiplier applied to LOD distance thresholds.  Values greater
        than 1.0 favour finer (higher-resolution) levels at a given
        camera distance; values less than 1.0 favour coarser levels.
        Default ``1.0`` (no bias).
    force_level : int or None
        When set, all bricks are assigned this 1-based LOD level,
        bypassing distance-based selection entirely.  ``None`` restores
        automatic selection.  Default ``None``.
    frustum_cull : bool
        When ``True``, bricks outside the camera frustum are skipped.
        When ``False``, all bricks in the scene are submitted regardless
        of visibility.  Default ``True``.
    """

    lod_bias: float = 1.0
    force_level: int | None = None
    frustum_cull: bool = True
