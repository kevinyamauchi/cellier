import numpy as np
import pygfx as gfx

from cellier.v2.render.lut_indirection._layout_2d import BlockLayout2D


class LutIndirectionManager2D:
    def __init__(self, base_layout: BlockLayout2D, n_levels: int) -> None:
        self._base_layout = base_layout
        self._n_levels = n_levels
        self.lut_data, self.lut_tex = build_lut_texture_2d(base_layout.grid_dims)


def build_lut_texture_2d(
    grid_dims: tuple[int, int],
) -> tuple[np.ndarray, gfx.Texture]:
    """Allocate CPU lut_data array and a gfx.Texture.

    Parameters
    ----------
    grid_dims : tuple[int, int]
        ``(gH, gW)`` -- finest level grid dimensions.

    Returns
    -------
    lut_data : np.ndarray
        Shape ``(gH, gW, 4)``, dtype float32.  Initialised to zeros
        (all tiles point to the reserved empty slot).
    lut_tex : gfx.Texture
        pygfx 2D texture wrapping ``lut_data``.
    """
    gh, gw = grid_dims
    lut_data = np.zeros((gh, gw, 4), dtype=np.float32)
    lut_tex = gfx.Texture(lut_data, dim=2)
    return lut_data, lut_tex
