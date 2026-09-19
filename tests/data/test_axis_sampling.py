"""Discrete versus continuous axes (the snap-vs-window annotation).

An image resolves a slice position by snapping to the nearest sample; a
geometry visual resolves it by containment in a window.  Those flip at
different instants -- a snap at the midpoint between samples, a window when
the position reaches the sample -- so without this annotation a graph's
markers lag the image it is drawn over.
"""

from uuid import uuid4

import numpy as np
import pytest

from cellier.data._axes import build_axes, data_coordinate_system
from cellier.data.graph._graph_memory_store import GraphMemoryStore
from cellier.transform import Axis

# -- the field ---------------------------------------------------------------


def test_the_default_is_continuous():
    """Unlike axis_type this has a default.

    A wrong axis_type selects the wrong plane silently; a wrong sampling
    degrades to the behaviour that predates the field -- a visible
    half-interval lag.
    """
    assert Axis(name="t", axis_type="time").sampling == "continuous"


def test_it_round_trips_through_json():
    axis = Axis(name="t", axis_type="time", sampling="discrete")
    assert Axis.model_validate_json(axis.model_dump_json()).sampling == "discrete"


def test_an_unknown_value_is_rejected():
    with pytest.raises(ValueError, match="sampling"):
        Axis(name="t", axis_type="time", sampling="sometimes")


# -- who sets it -------------------------------------------------------------


def test_a_store_keeps_the_sampling_its_system_declares():
    """The motivating case: t holds frame numbers, zyx hold positions.

    Per-axis rather than per-store, because a tracking graph is genuinely
    mixed.  The caller states it on the system; the store never overrides it.
    """
    axes = build_axes(
        ("t", "z", "y", "x"),
        types=("time", "space", "space", "space"),
        sampling=("discrete", "continuous", "continuous", "continuous"),
    )
    store = GraphMemoryStore(
        positions=np.zeros((2, 4)),
        edges=np.array([[0, 1]]),
        data_coordinate_systems=[data_coordinate_system(uuid4(), axes, "graph")],
    )
    axes = store.data_coordinate_systems[0].axes
    assert [axis.sampling for axis in axes] == [
        "discrete",
        "continuous",
        "continuous",
        "continuous",
    ]


# -- build_axes ---------------------------------------------------------------


def test_a_scalar_applies_to_every_axis():
    axes = build_axes(("z", "y", "x"), sampling="discrete")
    assert [axis.sampling for axis in axes] == ["discrete"] * 3


def test_a_sequence_is_per_axis():
    axes = build_axes(
        ("t", "z"),
        types=("time", "space"),
        sampling=("discrete", "continuous"),
    )
    assert [axis.sampling for axis in axes] == ["discrete", "continuous"]


def test_a_wrong_length_sequence_raises():
    with pytest.raises(ValueError, match="one entry per axis name"):
        build_axes(("z", "y", "x"), sampling=("discrete", "continuous"))
