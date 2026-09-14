"""The one round-half-up-and-clamp rule, shared across layers.

Both the render layer's ``round_world_to_voxel`` and the non-uniform axis
transform's nearest-neighbour lookup need "round half-up, then clamp to
``[0, size - 1]``".  Leaving them as two copies would let the two
conventions drift, and a drift here changes which plane a given world
position selects -- silently, and far from where the mistake was made.

This is a top-level module rather than a function in ``render/`` because
``transform/`` must not import ``render/``: the transform layer is
deliberately renderer-agnostic.  It sits alongside the other single-purpose
top-level modules (``logging.py``, ``types.py``).
"""

from __future__ import annotations

import numpy as np

__all__ = ["round_half_up", "round_half_up_clamped"]


def round_half_up(raw: float) -> int:
    """Round to the nearest index, half-up, with no clamping.

    ``floor(raw + 0.5)``.  Ties land on the higher index, which is
    deterministic and monotonic -- what a slider being dragged needs.

    Unclamped because not every consumer has a size to clamp to: a geometry
    store has vertices rather than a grid, so there is no extent to pin
    against, and a snapped position outside the data simply selects nothing.

    Parameters
    ----------
    raw : float
        A position already expressed in index units.

    Returns
    -------
    int
        The nearest index.
    """
    return int(np.floor(raw + 0.5))


def round_half_up_clamped(raw: float, size: int) -> int:
    """Round to the nearest index, half-up, then clamp to ``[0, size - 1]``.

    Round-half-up (``floor(raw + 0.5)``) is the rule consistent with the
    centre-at-integer convention used throughout cellier: index ``i`` is
    centred at integer ``i`` and spans ``[i - 0.5, i + 0.5)``.  Ties land on
    the higher index, which is deterministic and monotonic -- what a slider
    being dragged needs.

    Parameters
    ----------
    raw : float
        A position already expressed in index units.
    size : int
        The number of indices on the axis, used for clamping.

    Returns
    -------
    int
        An index in ``[0, size - 1]``.
    """
    return max(0, min(round_half_up(raw), size - 1))
