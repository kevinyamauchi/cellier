"""Per-scene render configuration for the RenderManager pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SceneRenderConfig:
    """Mutable render settings for one scene.

    The application retrieves the live config object from
    ``RenderManager.get_scene_config(scene_id)`` and mutates fields
    in-place.  The next call to ``trigger_update()`` (or any reslicing
    entry point) picks up the current values automatically.

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
