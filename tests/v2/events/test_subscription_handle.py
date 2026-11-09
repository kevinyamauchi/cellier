"""SubscriptionHandle and the bus's single removal path.

A handle removes its own subscription, and every route that removes a
subscription runs its ``on_remove`` cleanup exactly once (``clear`` skips it).
"""

from __future__ import annotations

import gc
from uuid import uuid4

import pytest

from cellier._state import AxisAlignedSelectionState, DimsState
from cellier.events import DimsChangedEvent, EventBus, SubscriptionHandle


def _dims_event(scene_id=None) -> DimsChangedEvent:
    return DimsChangedEvent(
        source_id=uuid4(),
        scene_id=scene_id or uuid4(),
        dims_state=DimsState(
            axis_labels=("z", "y", "x"),
            selection=AxisAlignedSelectionState(displayed_axes=(0, 1, 2)),
        ),
        displayed_axes_changed=False,
    )


class _Listener:
    def __init__(self) -> None:
        self.received: list = []

    def on_event(self, event) -> None:
        self.received.append(event)


def test_handle_unsubscribe_stops_the_callback():
    bus = EventBus()
    received: list = []
    handle = bus.subscribe(DimsChangedEvent, received.append)
    assert handle.active

    handle.unsubscribe()
    bus.emit(_dims_event())

    assert received == []
    assert not handle.active


def test_handle_unsubscribe_twice_is_a_no_op():
    bus = EventBus()
    removed: list = []
    handle = bus.subscribe(DimsChangedEvent, print, on_remove=lambda: removed.append(1))
    handle.unsubscribe()
    handle.unsubscribe()
    bus.unsubscribe(handle)

    assert removed == [1]


def test_a_callback_can_unsubscribe_itself():
    bus = EventBus()
    received: list = []

    def once(event) -> None:
        received.append(event)
        handle.unsubscribe()

    handle = bus.subscribe(DimsChangedEvent, once)
    first = _dims_event()
    bus.emit(first)
    bus.emit(_dims_event())

    assert received == [first]
    assert not handle.active


def test_a_subscription_removed_mid_emission_is_not_called():
    bus = EventBus()
    received: list = []
    bus.subscribe(DimsChangedEvent, lambda event: second.unsubscribe())
    second = bus.subscribe(DimsChangedEvent, received.append)

    bus.emit(_dims_event())

    assert received == []


@pytest.mark.parametrize("route", ["handle", "bus", "owner"])
def test_on_remove_runs_once_for_each_explicit_route(route):
    bus = EventBus()
    owner = uuid4()
    removed: list = []
    handle = bus.subscribe(
        DimsChangedEvent,
        print,
        owner_id=owner,
        on_remove=lambda: removed.append(route),
    )

    if route == "handle":
        handle.unsubscribe()
    elif route == "bus":
        bus.unsubscribe(handle)
    else:
        bus.unsubscribe_all(owner)
    # A second removal by another route does nothing more.
    bus.unsubscribe_all(owner)
    handle.unsubscribe()

    assert removed == [route]
    assert not handle.active


def test_on_remove_runs_when_a_dead_weak_callback_is_cleaned_up():
    bus = EventBus()
    removed: list = []
    listener = _Listener()
    handle = bus.subscribe(
        DimsChangedEvent,
        listener.on_event,
        weak=True,
        on_remove=lambda: removed.append(1),
    )
    del listener
    gc.collect()

    # Still registered until the next emission of its event type finds it.
    assert handle.active
    bus.emit(_dims_event())

    assert removed == [1]
    assert not handle.active


def test_clear_skips_on_remove_and_deactivates_handles():
    bus = EventBus()
    removed: list = []
    handle = bus.subscribe(DimsChangedEvent, print, on_remove=lambda: removed.append(1))

    bus.clear()
    handle.unsubscribe()

    assert removed == []
    assert not handle.active


def test_unsubscribe_all_finishes_when_a_cleanup_raises():
    bus = EventBus()
    owner = uuid4()
    removed: list = []

    def boom() -> None:
        raise RuntimeError("cleanup failed")

    first = bus.subscribe(DimsChangedEvent, print, owner_id=owner, on_remove=boom)
    second = bus.subscribe(
        DimsChangedEvent, print, owner_id=owner, on_remove=lambda: removed.append(1)
    )

    with pytest.raises(RuntimeError, match="cleanup failed"):
        bus.unsubscribe_all(owner)

    assert not first.active
    assert not second.active
    assert removed == [1]


def test_a_handle_does_not_keep_its_bus_alive():
    bus = EventBus()
    handle = bus.subscribe(DimsChangedEvent, print)
    del bus
    gc.collect()

    assert not handle.active
    handle.unsubscribe()  # must not raise


def test_a_handle_that_was_never_registered_is_inactive():
    bus = EventBus()
    handle = SubscriptionHandle(bus)

    assert not handle.active
    handle.unsubscribe()  # must not raise
    assert "inactive" in repr(handle)
