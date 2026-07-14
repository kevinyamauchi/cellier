# src/cellier/v2/visuals/_channel_appearance.py
from typing import Literal

from cmap import Colormap
from pydantic import ConfigDict

from cellier.visuals._base_visual import BaseAppearance


class ChannelAppearance(BaseAppearance):
    """Appearance parameters for one channel in a multichannel image visual.

    Parameters
    ----------
    color_map : cmap.Colormap
        Colormap applied to this channel's intensity values.
    clim : tuple[float, float]
        Contrast limits (min, max) for normalisation before colour-mapping.
    opacity : float
        Per-channel opacity in [0, 1].
    render_mode_3d : str
        Volume render mode for 3D: ``"mip"`` (default) or ``"iso"``.
    iso_threshold : float
        Isosurface threshold used when ``render_mode_3d == "iso"``.
        Default ``0.5``.
    visible : bool
        Inherited from ``BaseAppearance``. Default ``True``.

    Notes
    -----
    ``validate_assignment`` is enabled so that a value assigned to a field is
    coerced to the field's declared type (a ``color_map`` name string becomes a
    ``cmap.Colormap``; a ``clim`` list becomes a ``tuple``) and a malformed
    value raises ``pydantic.ValidationError``.  Widget-driven mutations arrive
    as JSON-native types (list/str), so this coercion is what lets the render
    layer consume them.
    """

    model_config = ConfigDict(validate_assignment=True)

    color_map: Colormap
    clim: tuple[float, float] = (0.0, 1.0)
    render_mode_3d: Literal["mip", "iso"] = "mip"
    iso_threshold: float = 0.5
    transparency_mode: Literal["blend", "add", "weighted_blend", "weighted_solid"] = (
        "add"
    )
