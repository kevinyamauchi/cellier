"""LUT parameter uniform buffer for the volume block shader.

The shader needs four vec3 values: ``block_size``, ``cache_size``,
``lut_size``, ``lut_offset``. Each vec3 is padded to 16 bytes (4xf32)
to satisfy WGSL uniform struct alignment rules.

All vec3 values use ``(x=W, y=H, z=D)`` axis convention to match
the WGSL shader coordinate system.
"""

import numpy as np
from pygfx.resources import Buffer

from block_volume.layout import BlockLayout

LUT_PARAMS_DTYPE = np.dtype(
    [
        ("block_size_x", "<f4"),
        ("block_size_y", "<f4"),
        ("block_size_z", "<f4"),
        ("_pad0", "<f4"),
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


def build_lut_params_buffer(layout: BlockLayout) -> Buffer:
    """Build the LUT parameter uniform buffer.

    Parameters
    ----------
    layout : BlockLayout
        Brick layout parameters.

    Returns
    -------
    buffer : Buffer
        Uniform buffer to be bound as ``u_lut_params`` in the shader.
    """
    gd, gh, gw = layout.grid_dims
    pd, ph, pw = layout.padded_shape
    bs = float(layout.block_size)

    # NOTE (L2): Must be a 0-d array, not shape (1,). A shape-(1,)
    # array causes pygfx to generate
    #   var<uniform> u_lut_params: array<LutParams, 1>
    # in WGSL, making direct field access (u_lut_params.block_size_x)
    # invalid.
    data = np.zeros((), dtype=LUT_PARAMS_DTYPE)
    data["block_size_x"] = bs
    data["block_size_y"] = bs
    data["block_size_z"] = bs
    data["cache_size_x"] = float(pw)
    data["cache_size_y"] = float(ph)
    data["cache_size_z"] = float(pd)
    data["lut_size_x"] = float(gw)
    data["lut_size_y"] = float(gh)
    data["lut_size_z"] = float(gd)
    data["lut_offset_x"] = 0.0
    data["lut_offset_y"] = 0.0
    data["lut_offset_z"] = 0.0

    return Buffer(data, force_contiguous=True)
