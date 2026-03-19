"""Backward-compatibility shim — import from image_block.core instead."""

from __future__ import annotations

from image_block.core.block_key import BlockKey as BrickKey  # deprecated alias
from image_block.core.tile_manager import TileManager, TileSlot

__all__ = ["BrickKey", "TileManager", "TileSlot"]
