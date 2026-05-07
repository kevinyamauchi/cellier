# src/cellier/v2/visuals/_channel_appearance.py
from typing import Literal

from cmap import Colormap

from cellier.v2.visuals._base_visual import BaseAppearance


class ChannelAppearance(BaseAppearance):
    """Appearance parameters for one channel in a multichannel image visual.

    Parameters
    ----------
    colormap : cmap.Colormap
        Colormap applied to this channel's intensity values.
    clim : tuple[float, float]
        Contrast limits (min, max) for normalisation before colour-mapping.
    opacity : float
        Per-channel opacity in [0, 1].
    render_mode_3d : str
        Volume render mode for 3D: ``"mip"`` (default) or ``"iso"``.
    visible : bool
        Inherited from ``BaseAppearance``. Default ``True``.
    """

    colormap: Colormap
    clim: tuple[float, float] = (0.0, 1.0)
    render_mode_3d: Literal["mip", "iso"] = "mip"
    transparency_mode: Literal["blend", "add", "weighted_blend", "weighted_solid"] = (
        "add"
    )
