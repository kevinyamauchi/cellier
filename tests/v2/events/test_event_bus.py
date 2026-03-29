"""Unit tests for EventBus invariants — no Qt, no GPU, no async."""

from __future__ import annotations

import gc
from uuid import uuid4

import pytest

from cellier.v2._state import AxisAlignedSelectionState, DimsState
from cellier.v2.events import (
    DimsChangedEvent,
    EventBus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dims_event(scene_id=None) -> DimsChangedEvent:
    return DimsChangedEvent(
        source_id=uuid4(),
        scene_id=scene_id or uuid4(),
        dims_state=DimsState(
            axis_labels=("z", "y", "x"),
            selection=AxisAlignedSelectionState(
                displayed_axes=(0, 1, 2), slice_indices={}
            ),
        ),
        displayed_axes_changed=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_emit_no_subscribers_is_noop():
    bus = EventBus()
    bus.emit(_make_dims_event())  # must not raise


def test_subscribe_no_filter_fires_always():
    bus = EventBus()
    received = []
    bus.subscribe(DimsChangedEvent, received.append)

    e1 = _make_dims_event()
    e2 = _make_dims_event()
    bus.emit(e1)
    bus.emit(e2)

    assert received == [e1, e2]


def test_subscribe_with_filter_fires_only_for_matching_entity():
    bus = EventBus()
    scene_a = uuid4()
    scene_b = uuid4()
    received = []
    bus.subscribe(DimsChangedEvent, received.append, entity_id=scene_a)

    bus.emit(_make_dims_event(scene_id=scene_a))
    bus.emit(_make_dims_event(scene_id=scene_b))

    assert len(received) == 1
    assert received[0].scene_id == scene_a


def test_unsubscribe_removes_entry():
    bus = EventBus()
    received = []
    handle = bus.subscribe(DimsChangedEvent, received.append)

    bus.emit(_make_dims_event())
    assert len(received) == 1

    bus.unsubscribe(handle)
    bus.emit(_make_dims_event())
    assert len(received) == 1  # no new events after unsubscribe


def test_unsubscribe_all_removes_all_owned():
    bus = EventBus()
    owner = uuid4()
    received = []
    bus.subscribe(DimsChangedEvent, received.append, owner_id=owner)
    bus.subscribe(DimsChangedEvent, received.append, owner_id=owner)

    bus.emit(_make_dims_event())
    assert len(received) == 2

    bus.unsubscribe_all(owner)
    bus.emit(_make_dims_event())
    assert len(received) == 2  # no new events after unsubscribe_all


def test_weak_subscription_dies_with_referent():
    """Dead weak entry is cleaned up on the next emit."""

    class _Handler:
        def __init__(self):
            self.received = []

        def handle(self, event):
            self.received.append(event)

    bus = EventBus()
    handler = _Handler()
    bus.subscribe(DimsChangedEvent, handler.handle, weak=True)

    bus.emit(_make_dims_event())
    assert len(handler.received) == 1

    del handler
    gc.collect()

    # Second emit — dead entry should be cleaned up, no error
    bus.emit(_make_dims_event())

    # Verify the dead entry was removed from the subscription list
    subs = bus._subs[DimsChangedEvent]
    assert all(s.is_alive() for s in subs)


def test_lambda_with_weak_raises():
    bus = EventBus()
    with pytest.raises(ValueError, match="lambda"):
        bus.subscribe(DimsChangedEvent, lambda e: None, weak=True)


def test_subscriber_exception_propagates_after_others_run():
    bus = EventBus()
    order = []

    def raises(event):
        order.append("raises")
        raise RuntimeError("boom")

    def second(event):
        order.append("second")

    bus.subscribe(DimsChangedEvent, raises)
    bus.subscribe(DimsChangedEvent, second)

    with pytest.raises(RuntimeError, match="boom"):
        bus.emit(_make_dims_event())

    assert order == ["raises", "second"]


def test_transform_changed_event_in_catalogue():
    from cellier.v2.events._events import CellierEventTypes, TransformChangedEvent

    assert TransformChangedEvent in CellierEventTypes.__args__
