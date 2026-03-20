"""Utility functions for converting cellier types to pygfx objects."""

from __future__ import annotations

import numpy as np
import pygfx as gfx


def cmap_to_gfx_colormap(colormap, n: int = 256) -> gfx.TextureMap:
    """Convert a ``cmap.Colormap`` to a pygfx ``TextureMap``.

    Parameters
    ----------
    colormap : cmap.Colormap
        A cmap colormap object (or anything callable with a float array
        that returns ``(N, 4)`` RGBA values in [0, 1]).
    n : int
        Number of sample points along the colourmap.

    Returns
    -------
    gfx.TextureMap
    """
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    rgba = np.asarray(colormap(t), dtype=np.float32)  # (n, 4)
    tex = gfx.Texture(rgba, dim=1)
    return gfx.TextureMap(tex, filter="linear")
