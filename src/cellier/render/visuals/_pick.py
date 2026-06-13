"""Shared helpers translating pygfx pick payloads to data coordinates.

All functions return continuous **level-0 data-array** coordinates in pygfx
``(x, y[, z])`` order, which equals displayed-axis order (pygfx local axis ``k``
maps to ``displayed_axes[k]``).  ``floor`` of each component yields the integer
voxel index: coordinates use the ``[i, i + 1)`` convention where voxel ``i``
spans ``[i, i + 1)`` and its center is ``i + 0.5``.

Keeping the math here — rather than in each GFX visual or in
``RenderManager._extract_pick_details`` — guarantees memory and multiscale,
image and labels, single- and multi-channel all share one rounding/clamping
convention so a pick decoded from any of them floors to the same index.
"""

from __future__ import annotations

# Pull the coordinate just inside the upper bound so an exact edge hit
# (continuous == size) floors to ``size - 1`` rather than out of range.
_EDGE_EPS = 1e-4


def memory_image_data_coordinate(
    pick_info: dict, ndim: int
) -> tuple[float, ...] | None:
    """Data coordinate for a memory image/labels node (real data texture).

    ``gfx.Image`` / ``gfx.Volume`` over the actual data already report the data
    index directly; we only convert pygfx's center-at-integer convention
    (``index`` + fractional ``pixel_coord`` / ``voxel_coord`` in ``[-0.5, 0.5)``)
    to the ``[i, i + 1)`` convention by adding ``0.5``.

    Parameters
    ----------
    pick_info : dict
        The pygfx pick payload.
    ndim : int
        2 for an image node, 3 for a volume node.
    """
    index = pick_info.get("index")
    if index is None:
        return None
    frac_key = "voxel_coord" if ndim == 3 else "pixel_coord"
    frac = pick_info.get(frac_key) or (0.0,) * ndim
    return tuple(float(index[k]) + float(frac[k]) + 0.5 for k in range(ndim))


def multiscale_volume_data_coordinate(
    pick_info: dict, norm_size, dataset_size
) -> tuple[float, ...] | None:
    """Data coordinate for a multiscale volume node (``NormSizedVolume``).

    The brick shader writes the surface hit as a centred-normalised position;
    ``NormSizedVolume._wgpu_get_pick_info`` decodes it to ``norm_pos``.  Map
    that back to level-0 voxels via the dataset's normalised half-extent
    (``norm_size``) and full voxel count (``dataset_size``), both in pygfx
    ``(x, y, z)`` order.

    Parameters
    ----------
    pick_info : dict
        The pygfx pick payload; must contain ``norm_pos``.
    norm_size, dataset_size : sequence of float
        Per-axis (x, y, z) normalised half-extent and level-0 voxel counts.
    """
    norm_pos = pick_info.get("norm_pos")
    if norm_pos is None:
        return None
    out = []
    for k in range(3):
        # norm_pos = (n - 0.5) * norm_size, with n the [0, 1] texture coord;
        # n * dataset_size is the continuous voxel position (center at i + 0.5).
        n = float(norm_pos[k]) / float(norm_size[k]) + 0.5
        size = float(dataset_size[k])
        out.append(min(max(n * size, 0.0), size - _EDGE_EPS))
    return tuple(out)


def multiscale_image_data_coordinate(
    pick_info: dict, level0_shape_hw, grid_dims_hw
) -> tuple[float, ...] | None:
    """Data coordinate for a multiscale 2-D image/labels node (tile proxy).

    The node is a ``gfx.Image`` over a *tile-grid* proxy texture (one texel per
    block), so pygfx's ``index`` is a tile, not a data pixel.  Reconstruct the
    continuous proxy-texel position (``index`` + ``pixel_coord``) and scale by
    ``level0 / grid`` (data pixels per proxy texel) to recover level-0 pixels in
    the ``[i, i + 1)`` convention (voxel ``i`` spans ``[i, i + 1)``, center
    ``i + 0.5``) — identical to :func:`memory_image_data_coordinate` and
    :func:`multiscale_volume_data_coordinate`.  The 2-D proxy node is placed
    center-at-integer (see ``_build_2d_node``), so a consumer converts this to a
    world position with a uniform ``- 0.5``, just like every other node.

    Parameters
    ----------
    pick_info : dict
        The pygfx pick payload.
    level0_shape_hw : tuple[int, int]
        Finest-level data shape ``(H, W)``.
    grid_dims_hw : tuple[int, int]
        Proxy tile-grid dimensions ``(gh, gw)``.
    """
    index = pick_info.get("index")
    if index is None:
        return None
    frac = pick_info.get("pixel_coord") or (0.0, 0.0)
    h0, w0 = float(level0_shape_hw[0]), float(level0_shape_hw[1])
    gh, gw = float(grid_dims_hw[0]), float(grid_dims_hw[1])
    # pygfx index[0] is texture-x (width/gw), index[1] is texture-y (height/gh).
    dx = (float(index[0]) + float(frac[0]) + 0.5) * (w0 / gw)
    dy = (float(index[1]) + float(frac[1]) + 0.5) * (h0 / gh)
    dx = min(max(dx, 0.0), w0 - _EDGE_EPS)
    dy = min(max(dy, 0.0), h0 - _EDGE_EPS)
    return (dx, dy)
