# src/cellier/v2/render/visuals/_multichannel_utils.py
"""Shared utilities for multichannel render visuals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygfx as gfx

from cellier.v2.render.visuals._image_memory import (
    _box_wireframe_positions,  # noqa: F401 (re-export)
    _make_colormap,  # (re-export)
    _pygfx_matrix,  # noqa: F401 (re-export)
    _rect_wireframe_positions,  # noqa: F401 (re-export)
    _transform_slice_indices,  # noqa: F401 (re-export)
)

if TYPE_CHECKING:
    from cellier.v2._state import DimsState
    from cellier.v2.data.image._image_requests import ChunkRequest
    from cellier.v2.visuals._channel_appearance import ChannelAppearance


def build_axis_selections_for_channel(
    dims_state: DimsState,
    store_shape: tuple[int, ...],
    channel_axis: int,
    channel_index: int,
) -> tuple[int | tuple[int, int], ...]:
    """Build axis_selections for a single-channel ChunkRequest.

    Displayed axes get ``(0, store_shape[ax])``; sliced axes get their integer
    index from ``dims_state.selection.slice_indices``; the channel axis always
    gets ``channel_index`` as a scalar int.

    Parameters
    ----------
    dims_state : DimsState
        Current dimension state.
    store_shape : tuple[int, ...]
        Full data-axis shape of the backing store.
    channel_axis : int
        Data-axis index for the channel dimension.
    channel_index : int
        Which channel to request along ``channel_axis``.

    Returns
    -------
    tuple[int | tuple[int, int], ...]
        One entry per data axis.
    """
    sel = dims_state.selection
    ndim = len(store_shape)
    result: list[int | tuple[int, int]] = []
    for ax in range(ndim):
        if ax == channel_axis:
            result.append(channel_index)
        elif ax in sel.displayed_axes:
            result.append((0, store_shape[ax]))
        else:
            result.append(sel.slice_indices[ax])
    return tuple(result)


def make_channel_group_2d(
    channels: dict[int, ChannelAppearance],
    max_channels: int,
    *,
    transparency_mode: str = "add",
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
    material.map = _make_colormap(appearance.colormap)
    material.opacity = appearance.opacity
    material.alpha_mode = appearance.transparency_mode
    node.visible = appearance.visible


def apply_channel_appearance_3d(
    node: gfx.Volume,
    appearance: ChannelAppearance,
) -> None:
    """Apply a ChannelAppearance to a 3D Volume node's material.

    Parameters
    ----------
    node : gfx.Volume
        Target volume node.
    appearance : ChannelAppearance
        Appearance settings to apply.
    """
    material = node.material
    material.clim = appearance.clim
    material.map = _make_colormap(appearance.colormap)
    material.opacity = appearance.opacity
    material.alpha_mode = appearance.transparency_mode
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
