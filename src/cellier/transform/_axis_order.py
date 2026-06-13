"""Axis selection and permutation primitives for nD value sequences.

These helpers operate on plain tuples/sequences (e.g. shape tuples,
scale vectors, translation vectors).  The analogous operations on
:class:`AffineTransform` live on the transform itself as
``select_axes`` and ``swap_axes``.

Conventions
-----------
- ``select_axes`` picks a subset of entries from a per-data-axis
  sequence.  The result preserves the order in ``axes`` and is therefore
  still in *data axis order* over the selected subset.
- ``swap_axes`` reorders an entire sequence by an explicit permutation.
  Use this to convert between data-axis order and display-axis order.
  The data-to-display permutation is *not* hardcoded here -- it is
  always supplied by the caller as a literal tuple, so the assumption
  is visible at the call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence


T = TypeVar("T")


def select_axes(values: Sequence[T], axes: tuple[int, ...]) -> tuple[T, ...]:
    """Select entries of ``values`` corresponding to ``axes``.

    The result is in the order given by ``axes`` (typically still data
    axis order over the selected subset).

    Parameters
    ----------
    values : Sequence[T]
        Per-data-axis sequence (e.g. a shape tuple or scale vector).
    axes : tuple[int, ...]
        Indices into ``values`` to keep.

    Returns
    -------
    tuple[T, ...]
        ``tuple(values[a] for a in axes)``.
    """
    return tuple(values[a] for a in axes)


def swap_axes(values: Sequence[T], permutation: tuple[int, ...]) -> tuple[T, ...]:
    """Reorder ``values`` by an explicit permutation.

    ``permutation[i]`` is the source index whose value becomes output
    index ``i``.  ``permutation`` must be a permutation of
    ``range(len(values))``.

    Parameters
    ----------
    values : Sequence[T]
        Sequence to reorder.
    permutation : tuple[int, ...]
        Permutation of ``range(len(values))``.

    Returns
    -------
    tuple[T, ...]
        ``tuple(values[p] for p in permutation)``.

    Raises
    ------
    ValueError
        If ``permutation`` is not a permutation of ``range(len(values))``.
    """
    n = len(values)
    if len(permutation) != n or sorted(permutation) != list(range(n)):
        raise ValueError(
            f"permutation must be a permutation of range({n}), got {permutation}"
        )
    return tuple(values[p] for p in permutation)
