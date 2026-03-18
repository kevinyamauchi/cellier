"""LUT parameter and block-scales uniform buffers for the volume block shader.

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
        # vol_size carries the real padded voxel dimensions of the volume.
        # Used by fs_main so the raycaster step/gradient calculation is
        # correct even when the proxy texture uses grid dims (1 texel/brick)
        # instead of the full voxel resolution.
        # (Repurposed from lut_offset_x/y/z which were always zero.)
        ("vol_size_x", "<f4"),
        ("vol_size_y", "<f4"),
        ("vol_size_z", "<f4"),
        ("_pad3", "<f4"),
    ]
)


def build_lut_params_buffer(
    base_layout: BlockLayout,
    cache_info: CacheInfo,
    proxy_voxels_per_brick: float | None = None,
) -> Buffer:
    """Build the LUT parameter uniform buffer.

    Parameters
    ----------
    base_layout : BlockLayout
        Layout of the finest (level 1) resolution.
    cache_info : CacheInfo
        Cache sizing metadata.
    proxy_voxels_per_brick : float or None
        When the proxy texture uses grid dims (1 texel per brick), pass
        ``block_size`` here.  All spatial LUT uniforms are normalised to
        **grid units** (1 unit = 1 brick) so the shader's LUT look-up
        (brick index, cache coord) is consistent with the data-space
        coordinates produced by vs_main.

        ``vol_size_*`` always carries the real padded voxel extent so
        ``fs_main`` can calibrate step density correctly.

    Returns
    -------
    buffer : Buffer
        Uniform buffer bound as ``u_lut_params`` in the shader.
    """
    gd, gh, gw = base_layout.grid_dims
    pd, ph, pw = base_layout.padded_shape  # real voxel extent
    bs = float(base_layout.block_size)
    ov = float(cache_info.overlap)

    data = np.zeros((), dtype=LUT_PARAMS_DTYPE)

    if proxy_voxels_per_brick is not None:
        # Grid-unit coordinate space: 1 proxy texel = 1 brick = p voxels.
        # Normalise block_size, overlap, and cache_size by p.
        p = float(proxy_voxels_per_brick)
        ov_u    = ov / p                        # overlap in grid units
        padded_u = 1.0 + 2.0 * ov_u            # padded brick size, grid units
        gs      = float(cache_info.grid_side)
        cs_u    = gs * padded_u                 # cache extent per axis, grid units

        data["block_size_x"] = 1.0
        data["block_size_y"] = 1.0
        data["block_size_z"] = 1.0
        data["overlap"]      = ov_u
        data["cache_size_x"] = cs_u
        data["cache_size_y"] = cs_u
        data["cache_size_z"] = cs_u
    else:
        # Voxel-space: proxy has full padded_shape dimensions.
        cd, ch, cw = cache_info.cache_shape
        data["block_size_x"] = bs
        data["block_size_y"] = bs
        data["block_size_z"] = bs
        data["overlap"]      = ov
        data["cache_size_x"] = float(cw)
        data["cache_size_y"] = float(ch)
        data["cache_size_z"] = float(cd)

    data["lut_size_x"]   = float(gw)
    data["lut_size_y"]   = float(gh)
    data["lut_size_z"]   = float(gd)
    # Real padded voxel dimensions for step-density calibration in fs_main.
    data["vol_size_x"]   = float(pw)
    data["vol_size_y"]   = float(ph)
    data["vol_size_z"]   = float(pd)

    return Buffer(data, force_contiguous=True)


# ── Block scales ────────────────────────────────────────────────────────

MAX_LEVELS = 10

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
