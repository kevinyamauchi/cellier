"""LUT parameter and block-scales uniform buffers for the volume block shader.

Phase 1 adds ``BLOCK_SCALES_DTYPE`` carrying 10 entries of ``vec4<f32>``
that map each LOD level to a per-axis compression factor.

All vec3 values use ``(x=W, y=H, z=D)`` axis convention to match
the WGSL shader coordinate system.
"""

from __future__ import annotations

import numpy as np
from pygfx.resources import Buffer

from block_volume.cache import CacheInfo
from block_volume.layout import BlockLayout

# ── LUT params ──────────────────────────────────────────────────────────

LUT_PARAMS_DTYPE = np.dtype(
    [
        ("block_size_x", "<f4"),
        ("block_size_y", "<f4"),
        ("block_size_z", "<f4"),
        ("overlap", "<f4"),
        ("cache_size_x", "<f4"),
        ("cache_size_y", "<f4"),
        ("cache_size_z", "<f4"),
        ("_pad1", "<f4"),
        ("lut_size_x", "<f4"),
        ("lut_size_y", "<f4"),
        ("lut_size_z", "<f4"),
        ("_pad2", "<f4"),
        ("lut_offset_x", "<f4"),
        ("lut_offset_y", "<f4"),
        ("lut_offset_z", "<f4"),
        ("_pad3", "<f4"),
    ]
)


def build_lut_params_buffer(
    base_layout: BlockLayout,
    cache_info: CacheInfo,
) -> Buffer:
    """Build the LUT parameter uniform buffer.

    Parameters
    ----------
    base_layout : BlockLayout
        Layout of the finest (level 1) resolution.
    cache_info : CacheInfo
        Cache sizing metadata.

    Returns
    -------
    buffer : Buffer
        Uniform buffer bound as ``u_lut_params`` in the shader.
    """
    gd, gh, gw = base_layout.grid_dims
    cd, ch, cw = cache_info.cache_shape
    bs = float(base_layout.block_size)

    # NOTE (L2): Must be 0-d array, not shape (1,).
    data = np.zeros((), dtype=LUT_PARAMS_DTYPE)
    data["block_size_x"] = bs
    data["block_size_y"] = bs
    data["block_size_z"] = bs
    data["overlap"] = float(cache_info.overlap)
    data["cache_size_x"] = float(cw)
    data["cache_size_y"] = float(ch)
    data["cache_size_z"] = float(cd)
    data["lut_size_x"] = float(gw)
    data["lut_size_y"] = float(gh)
    data["lut_size_z"] = float(gd)
    data["lut_offset_x"] = 0.0
    data["lut_offset_y"] = 0.0
    data["lut_offset_z"] = 0.0

    return Buffer(data, force_contiguous=True)


# ── Block scales ────────────────────────────────────────────────────────

MAX_LEVELS = 10

# Each level is a separate vec4 field so that pygfx's uniform struct
# generator produces valid WGSL (it cannot handle 2D array shapes).
BLOCK_SCALES_DTYPE = np.dtype(
    [(f"scale_{i}", "<f4", (4,)) for i in range(MAX_LEVELS)]
)


def build_block_scales_buffer(n_levels: int) -> Buffer:
    """Build the block-scales uniform buffer.

    Level 0 is reserved (all zeros — renders black).
    Level *k* (1-indexed, where 1 = finest) gets
    ``sj = (1 / 2**(k-1),) * 3``.

    Parameters
    ----------
    n_levels : int
        Number of LOD levels (e.g. 3).

    Returns
    -------
    buffer : Buffer
        Uniform buffer bound as ``u_block_scales`` in the shader.
    """
    data = np.zeros((), dtype=BLOCK_SCALES_DTYPE)

    for k in range(1, min(n_levels + 1, MAX_LEVELS)):
        s = 1.0 / (2 ** (k - 1))
        data[f"scale_{k}"][0] = s  # x
        data[f"scale_{k}"][1] = s  # y
        data[f"scale_{k}"][2] = s  # z
        # [3] stays 0.0, unused

    return Buffer(data, force_contiguous=True)
