# src/cellier/v2/render/visuals/_multichannel_utils.py
"""Shared utilities for multichannel render visuals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygfx as gfx

from cellier.render.visuals._image_memory import (
    _VOLUME_MATERIALS,
    # (re-export)
    _make_colormap,  # (re-export)
)

if TYPE_CHECKING:
    from cellier.data.image._image_requests import ChunkRequest
    from cellier.visuals._channel_appearance import ChannelAppearance


def make_channel_group_2d(
    channels: dict[int, ChannelAppearance],
    max_channels: int,
    *,
    transparency_mode: str = "add",
    interpolation: str = "nearest",
    pick_write: bool = True,
) -> tuple[gfx.Group, list[gfx.Image]]:
    """Allocate a 2D node pool as a Group containing ``max_channels`` Image nodes.

    All nodes start hidden. Returns ``(group, pool)`` where ``pool[slot]`` is
    the ``gfx.Image`` for that slot. Slots are assigned in dict-iteration order.

    Parameters
    ----------
    channels : dict[int, ChannelAppearance]
        Initial channel appearances.
    max_channels : int
        Total number of pre-allocated pool slots.
    transparency_mode : str
        pygfx alpha_mode for the pool nodes. Default ``"add"``.
    interpolation : str
        Sampler filter for all pool nodes. Default ``"nearest"``.
    pick_write : bool
        Whether pool materials write to the pick buffer.  Default ``True``.

    Returns
    -------
    tuple[gfx.Group, list[gfx.Image]]
    """
    alpha_mode = transparency_mode
    pool: list[gfx.Image] = []
    group = gfx.Group()

    for _slot in range(max_channels):
        placeholder = np.zeros((1, 1, 1), dtype=np.float32)
        tex = gfx.Texture(placeholder, dim=2, format="1xf4")
        node = gfx.Image(
            gfx.Geometry(grid=tex),
            gfx.ImageBasicMaterial(
                clim=(0.0, 1.0),
                map=gfx.cm.viridis,
                alpha_mode=alpha_mode,
                interpolation=interpolation,
                pick_write=pick_write,
            ),
        )
        node.visible = False
        group.add(node)
        pool.append(node)

    return group, pool


def make_channel_group_3d(
    channels: dict[int, ChannelAppearance],
    max_channels: int,
    *,
    transparency_mode: str = "add",
    interpolation: str = "nearest",
    pick_write: bool = True,
) -> tuple[gfx.Group, list[gfx.Volume]]:
    """Allocate a 3D node pool as a Group containing ``max_channels`` Volume nodes.

    Parameters
    ----------
    channels : dict[int, ChannelAppearance]
        Initial channel appearances.
    max_channels : int
        Total number of pre-allocated pool slots.
    transparency_mode : str
        pygfx alpha_mode for the pool nodes. Default ``"add"``.
    interpolation : str
        Sampler filter for all pool nodes. Default ``"nearest"``.
    pick_write : bool
        Whether pool materials write to the pick buffer.  Default ``True``.

    Returns
    -------
    tuple[gfx.Group, list[gfx.Volume]]
    """
    alpha_mode = transparency_mode
    pool: list[gfx.Volume] = []
    group = gfx.Group()

    for _slot in range(max_channels):
        placeholder = np.zeros((2, 2, 2), dtype=np.float32)
        tex = gfx.Texture(placeholder, dim=3, format="1xf4")
        node = gfx.Volume(
            gfx.Geometry(grid=tex),
            gfx.VolumeMipMaterial(
                clim=(0.0, 1.0),
                map=gfx.cm.viridis,
                alpha_mode=alpha_mode,
                interpolation=interpolation,
                pick_write=pick_write,
            ),
        )
        node.visible = False
        group.add(node)
        pool.append(node)

    return group, pool


def apply_channel_appearance_2d(
    node: gfx.Image,
    appearance: ChannelAppearance,
) -> None:
    """Apply a ChannelAppearance to a 2D Image node's material.

    Parameters
    ----------
    node : gfx.Image
        Target image node.
    appearance : ChannelAppearance
        Appearance settings to apply.
    """
    material = node.material
    material.clim = appearance.clim
    material.map = _make_colormap(appearance.color_map)
    material.opacity = appearance.opacity
    material.alpha_mode = appearance.transparency_mode
    node.visible = appearance.visible


def apply_channel_appearance_3d(
    node: gfx.Volume,
    appearance: ChannelAppearance,
) -> None:
    """Apply a ChannelAppearance to a 3D Volume node's material.

    If ``appearance.render_mode_3d`` requires a different volume material class
    than the node currently holds (pool nodes default to MIP and are reused
    across channels), the material is swapped in place, preserving the node's
    sampler ``interpolation`` and ``pick_write`` settings.

    Parameters
    ----------
    node : gfx.Volume
        Target volume node.
    appearance : ChannelAppearance
        Appearance settings to apply.
    """
    desired_cls = _VOLUME_MATERIALS[appearance.render_mode_3d]
    material = node.material
    if not isinstance(material, desired_cls):
        old = material
        material = desired_cls(
            clim=appearance.clim,
            map=_make_colormap(appearance.color_map),
            alpha_mode=appearance.transparency_mode,
            interpolation=old.interpolation,
            pick_write=old.pick_write,
        )
        node.material = material
    material.clim = appearance.clim
    material.map = _make_colormap(appearance.color_map)
    material.opacity = appearance.opacity
    material.alpha_mode = appearance.transparency_mode
    if appearance.render_mode_3d == "iso":
        material.threshold = appearance.iso_threshold
    node.visible = appearance.visible


def channel_index_from_request(
    request: ChunkRequest,
    channel_axis: int,
) -> int:
    """Extract the channel index from a ChunkRequest's axis_selections.

    Parameters
    ----------
    request : ChunkRequest
        Request whose ``axis_selections`` to inspect.
    channel_axis : int
        Data-axis index for the channel dimension.

    Returns
    -------
    int
    """
    val = request.axis_selections[channel_axis]
    return int(val)
