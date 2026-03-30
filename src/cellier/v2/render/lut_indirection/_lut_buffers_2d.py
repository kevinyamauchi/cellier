"""LUT parameter and block-scales uniform buffers for the 2D image block shader.

All vec2 values use ``(x=W, y=H)`` axis convention to match WGSL.
Uniform structs are padded to vec4 boundaries per WGSL alignment rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pygfx.resources import Buffer

if TYPE_CHECKING:
    from cellier.v2.render.block_cache._cache_parameters_2d import (
        BlockCacheParameters2D,
    )
    from cellier.v2.render.lut_indirection._layout_2d import BlockLayout2D

# ---- LUT params --------------------------------------------------------

LUT_PARAMS_DTYPE = np.dtype(
    [
        ("block_size_x", "<f4"),
        ("block_size_y", "<f4"),
        ("overlap", "<f4"),
        ("_pad0", "<f4"),
        ("cache_size_x", "<f4"),
        ("cache_size_y", "<f4"),
        ("_pad1", "<f4"),
        ("_pad2", "<f4"),
        ("lut_size_x", "<f4"),
        ("lut_size_y", "<f4"),
        ("_pad3", "<f4"),
        ("_pad4", "<f4"),
        ("vol_size_x", "<f4"),
        ("vol_size_y", "<f4"),
        ("_pad5", "<f4"),
        ("_pad6", "<f4"),
    ]
)


def build_lut_params_buffer_2d(
    base_layout: BlockLayout2D,
    cache_info: BlockCacheParameters2D,
) -> Buffer:
    """Build the LUT parameter uniform buffer.

    The proxy texture uses grid dims (1 texel per tile), so spatial
    uniforms are normalised to grid units (1 unit = 1 tile).

    Parameters
    ----------
    base_layout : BlockLayout2D
        Layout of the finest (level 1) resolution.
    cache_info : BlockCacheParameters2D
        Cache sizing metadata.

    Returns
    -------
    buffer : Buffer
        Uniform buffer bound as ``u_lut_params`` in the shader.
    """
    gh, gw = base_layout.grid_dims
    bs = float(base_layout.block_size)
    ov = float(cache_info.overlap)
    gs = float(cache_info.grid_side)

    # Grid-unit coordinate space: 1 proxy texel = 1 tile.
    # block_size in grid units = 1.0
    # overlap in grid units = overlap_pixels / block_size
    ov_u = ov / bs
    padded_u = 1.0 + 2.0 * ov_u
    cs_u = gs * padded_u  # cache extent per axis in grid units

    ph, pw = base_layout.padded_shape

    # NOTE: Must be 0-d array, not shape (1,).
    data = np.zeros((), dtype=LUT_PARAMS_DTYPE)
    data["block_size_x"] = 1.0
    data["block_size_y"] = 1.0
    data["overlap"] = ov_u
    data["cache_size_x"] = cs_u
    data["cache_size_y"] = cs_u
    data["lut_size_x"] = float(gw)
    data["lut_size_y"] = float(gh)
    data["vol_size_x"] = float(pw)
    data["vol_size_y"] = float(ph)

    return Buffer(data, force_contiguous=True)


# ---- Block scales -------------------------------------------------------

MAX_LEVELS = 10

# Each level is a separate vec4 field.
BLOCK_SCALES_DTYPE = np.dtype([(f"scale_{i}", "<f4", (4,)) for i in range(MAX_LEVELS)])


def build_block_scales_buffer_2d(
    level_scale_vecs_data: list[np.ndarray] | None = None,
    n_levels: int | None = None,
) -> Buffer:
    """Build the block-scales uniform buffer.

    Level 0 is reserved (all zeros -- renders black).
    Level k (1-indexed, 1 = finest) gets ``1 / scale_factor`` per axis.

    Input vectors are in data-axis order ``(sy, sx)``; shader
    fields ``[0], [1]`` are ``(x=W, y=H)`` so the assignment reverses.

    Parameters
    ----------
    level_scale_vecs_data : list[np.ndarray] or None
        Per-level scale vectors in 2D data order ``(sy, sx)``.
    n_levels : int or None
        Fallback: number of LOD levels for power-of-2 scales.

    Returns
    -------
    buffer : Buffer
        Uniform buffer bound as ``u_block_scales`` in the shader.
    """
    data = np.zeros((), dtype=BLOCK_SCALES_DTYPE)

    if level_scale_vecs_data is not None:
        n = len(level_scale_vecs_data)
        for k in range(1, min(n + 1, MAX_LEVELS)):
            sv = level_scale_vecs_data[k - 1]  # data order: [sy, sx]
            # shader x = W = data axis 1 (sx)
            data[f"scale_{k}"][0] = 1.0 / float(sv[1])
            # shader y = H = data axis 0 (sy)
            data[f"scale_{k}"][1] = 1.0 / float(sv[0])
            data[f"scale_{k}"][2] = 0.0  # unused z
    elif n_levels is not None:
        for k in range(1, min(n_levels + 1, MAX_LEVELS)):
            s = 1.0 / (2 ** (k - 1))
            data[f"scale_{k}"][0] = s  # x
            data[f"scale_{k}"][1] = s  # y
            data[f"scale_{k}"][2] = 0.0  # unused z

    return Buffer(data, force_contiguous=True)
