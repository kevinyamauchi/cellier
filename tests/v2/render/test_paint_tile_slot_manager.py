"""Unit tests for PaintTileSlotManager."""

from __future__ import annotations

import pytest

from cellier.v2.render.visuals._paint_tile_slot_manager import (
    PaintTileSlotManager,
)


def test_first_allocation_returns_zero() -> None:
    m = PaintTileSlotManager(max_slots=4)
    assert m.get_or_allocate((0, 0)) == 0


def test_repeated_allocation_returns_same_slot() -> None:
    m = PaintTileSlotManager(max_slots=4)
    slot1 = m.get_or_allocate((1, 2))
    slot2 = m.get_or_allocate((1, 2))
    assert slot1 == slot2


def test_distinct_tiles_get_distinct_slots() -> None:
    m = PaintTileSlotManager(max_slots=4)
    s1 = m.get_or_allocate((0, 0))
    s2 = m.get_or_allocate((0, 1))
    s3 = m.get_or_allocate((1, 0))
    assert {s1, s2, s3} == {0, 1, 2}


def test_pool_exhaustion_returns_none_and_warns() -> None:
    m = PaintTileSlotManager(max_slots=2)
    assert m.get_or_allocate((0, 0)) == 0
    assert m.get_or_allocate((0, 1)) == 1
    assert m.get_or_allocate((0, 2)) is None
    assert m.exhausted is True
    # Existing tiles still map to their slots.
    assert m.get_or_allocate((0, 0)) == 0


def test_clear_resets_and_clears_exhaustion() -> None:
    m = PaintTileSlotManager(max_slots=1)
    m.get_or_allocate((0, 0))
    m.get_or_allocate((1, 1))  # exhausts
    assert m.exhausted is True
    m.clear()
    assert m.n_allocated == 0
    assert m.exhausted is False
    # Pool is freshly empty.
    assert m.get_or_allocate((9, 9)) == 0


def test_get_does_not_allocate() -> None:
    m = PaintTileSlotManager(max_slots=4)
    assert m.get((0, 0)) is None
    m.get_or_allocate((0, 0))
    assert m.get((0, 0)) == 0


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_max_slots(bad: int) -> None:
    with pytest.raises(ValueError):
        PaintTileSlotManager(max_slots=bad)
