"""Label colormap GPU resource builders."""

from __future__ import annotations

import warnings

import numpy as np
import pygfx as gfx
import wgpu

_MAX_DIRECT_ENTRIES = 65536

LABEL_PARAMS_DTYPE = np.dtype(
    [
        ("background_label", np.int32),
        ("salt", np.uint32),
        ("n_entries", np.uint32),
        ("_pad", np.uint32),
    ]
)


def build_direct_lut_textures(
    color_dict: dict[int, tuple[float, float, float, float]],
) -> tuple[gfx.Texture, gfx.Texture, int]:
    """Build GPU textures for direct-mode colormap binary search.

    Returns (keys_tex, colors_tex, n_entries).
    keys_tex  : r32sint 2D texture, shape (H=1, W=n, C=1) — sorted int32 IDs
    colors_tex: rgba32float 2D texture, shape (H=1, W=n, C=4) — RGBA per entry
    n_entries : int — number of entries (0 for empty dict)
    """
    n_raw = len(color_dict)
    if n_raw > _MAX_DIRECT_ENTRIES:
        warnings.warn(
            f"color_dict has {n_raw} entries, which exceeds the maximum of "
            f"{_MAX_DIRECT_ENTRIES}. Only the {_MAX_DIRECT_ENTRIES} entries "
            f"with the smallest label IDs will be used.",
            stacklevel=3,
        )
    keys_sorted = sorted(color_dict.keys())[:_MAX_DIRECT_ENTRIES]
    n = len(keys_sorted)

    if n == 0:
        key_data = np.array([[0]], dtype=np.int32).reshape(1, 1, 1)
        color_data = np.zeros((1, 1, 4), dtype=np.float32)
        return (
            gfx.Texture(key_data, dim=2, format="1xi4"),
            gfx.Texture(color_data, dim=2, format="4xf4"),
            0,
        )

    # pygfx 2D texture shape: (H, W, C)
    key_data = np.array(keys_sorted, dtype=np.int32).reshape(1, n, 1)
    color_data = np.array(
        [color_dict[k] for k in keys_sorted], dtype=np.float32
    ).reshape(1, n, 4)

    keys_tex = gfx.Texture(key_data, dim=2, format="1xi4")
    colors_tex = gfx.Texture(color_data, dim=2, format="4xf4")
    return keys_tex, colors_tex, n


def build_label_params_buffer(
    background_label: int,
    salt: int,
    n_entries: int,
) -> gfx.Buffer:
    """Build the u_label_params uniform buffer (16 bytes, aligned)."""
    params = np.zeros((), dtype=LABEL_PARAMS_DTYPE)
    params["background_label"] = np.int32(background_label)
    params["salt"] = np.uint32(salt & 0xFFFFFFFF)
    params["n_entries"] = np.uint32(n_entries)
    return gfx.Buffer(
        params, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
    )
