"""Temporal accumulation effect pass for noise reduction in volume rendering.

Blends successive jittered frames via exponential moving average (EMA).
When the camera is still, noise decreases by ~sqrt(N) after N frames.
When the camera moves, call ``reset()`` to discard history and restart.
"""

from __future__ import annotations

from typing import ClassVar

import wgpu
from pygfx.renderers.wgpu.engine.effectpasses import EffectPass, FullQuadPass
from pygfx.renderers.wgpu.engine.shared import get_shared


class _BlendPass(FullQuadPass):
    """Inner full-quad pass that blends the current frame with history."""

    uniform_type: ClassVar[dict] = {"weight": "f4"}

    wgsl = """
        @fragment
        fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
            let current = textureSample(colorTex, texSampler, varyings.texCoord);
            let history = textureSample(historyTex, texSampler, varyings.texCoord);
            return mix(history, current, u_effect.weight);
        }
    """


class TemporalAccumulationPass(EffectPass):
    """Post-processing pass that accumulates jittered frames over time.

    Insert as the first entry in ``renderer.effect_passes`` so it
    operates on the raw raymarched output before anti-aliasing.

    Parameters
    ----------
    alpha : float
        Minimum blend weight for the current frame.  During warm-up the
        weight is ``1 / (frame_count + 1)``; once that falls below
        ``alpha`` the weight clamps to ``alpha``.  Lower values give
        smoother steady-state but slower convergence.  Default 0.1
        (steady state reached in ~10 still frames).
    """

    # Fallback shader (used by the EffectPass base machinery if it ever
    # needs to compile a pipeline for *this* object — in practice the
    # blend is done by _BlendPass and we only copy to the target).
    wgsl = """
        @fragment
        fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
            return textureSample(colorTex, texSampler, varyings.texCoord);
        }
    """

    def __init__(self, *, alpha: float = 0.1) -> None:
        super().__init__()
        self._alpha = float(alpha)
        self._frame_count: int = 0
        self._history_index: int = 0
        self._history_textures: list[wgpu.GPUTexture] = []
        self._history_views: list[wgpu.GPUTextureView] = []
        self._current_size: tuple[int, int] | None = None
        self._blend_pass = _BlendPass()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Discard accumulated history.

        The next frame's blend weight becomes 1.0, so history is
        immediately overwritten.  No GPU memory is freed or cleared.
        """
        self._frame_count = 0

    @property
    def alpha(self) -> float:
        """Minimum blend weight for the current frame (EMA floor).

        Lower values give smoother steady-state at the cost of slower
        convergence after a reset.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = float(value)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(
        self,
        command_encoder: wgpu.GPUCommandEncoder,
        color_tex: wgpu.GPUTextureView,
        depth_tex: wgpu.GPUTextureView | None,
        target_tex: wgpu.GPUTextureView,
    ) -> None:
        # --- Lazy (re)allocation on size change ---
        w, h = color_tex.texture.size[:2]
        if (w, h) != self._current_size:
            self._reallocate_history(w, h, color_tex.texture.format)
            self._current_size = (w, h)
            self.reset()

        # --- Compute EMA weight ---
        # Warm-up: running average until weight would drop below alpha,
        # then clamp to alpha for a stable steady-state convergence rate.
        weight = max(1.0 / (self._frame_count + 1), self._alpha)

        # --- Blend current frame with history ---
        read_view = self._history_views[1 - self._history_index]
        write_view = self._history_views[self._history_index]

        self._blend_pass._uniform_data["weight"] = weight
        self._blend_pass.render(
            command_encoder,
            colorTex=color_tex,
            historyTex=read_view,
            targetTex=write_view,
        )

        # --- Copy blended result to downstream target via render pass ---
        # We use the base EffectPass (CopyPass-style) to avoid requiring
        # COPY_DST on the blender's target texture.
        super().render(command_encoder, write_view, depth_tex, target_tex)

        # --- Advance state ---
        self._history_index = 1 - self._history_index
        self._frame_count += 1

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reallocate_history(self, width: int, height: int, fmt: str) -> None:
        """Create (or recreate) the two ping-pong history textures."""
        device = get_shared().device
        usage = wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.RENDER_ATTACHMENT

        self._history_textures = []
        self._history_views = []
        for _ in range(2):
            tex = device.create_texture(
                size=(width, height, 1),
                format=fmt,
                usage=usage,
                dimension="2d",
            )
            self._history_textures.append(tex)
            self._history_views.append(tex.create_view())
