"""Unified block key replacing TileKey (2D) and BrickKey (3D)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockKey:
    """Identifier for a block (tile or brick) at a specific LOAD level.

    Grid coordinates are in numpy axis order: z (depth), y (height), x
    (width).  For 2D tiles gz is always 0.

    Attributes
    ----------
    level : int
        1-indexed LOAD level (1 = finest).
    gz : int
        Grid z-index (depth axis).  Always 0 for 2D tiles.
    gy : int
        Grid y-index (height axis).
    gx : int
        Grid x-index (width axis).
    """

    level: int
    gz: int
    gy: int
    gx: int
