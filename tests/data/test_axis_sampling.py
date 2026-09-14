"""Discrete versus continuous axes (the snap-vs-window annotation).

An image resolves a slice position by snapping to the nearest sample; a
geometry visual resolves it by containment in a window.  Those flip at
different instants -- a snap at the midpoint between samples, a window when
the position reaches the sample -- so without this annotation a graph's
markers lag the image it is drawn over.
"""

import numpy as np
import pytest

from cellier.data._axes import build_axes
from cellier.data.graph._graph_memory_store import GraphMemoryStore
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.label._label_memory_store import LabelMemoryStore
from cellier.data.lines._lines_memory_store import LinesMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
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


@pytest.mark.parametrize(
    "store",
    [
        ImageMemoryStore(
            data=np.zeros((2, 3, 4), dtype=np.float32), axis_names=("z", "y", "x")
        ),
        LabelMemoryStore(
            data=np.zeros((2, 3, 4), dtype=np.int32), axis_names=("z", "y", "x")
        ),
    ],
    ids=["image", "labels"],
)
def test_a_gridded_store_declares_itself_discrete(store):
    """A voxel grid is sample-indexed by construction, so callers never say so."""
    axes = store.data_coordinate_systems[0].axes
    assert [axis.sampling for axis in axes] == ["discrete"] * 3


@pytest.mark.parametrize(
    "store",
    [
        PointsMemoryStore(positions=np.zeros((2, 3)), axis_names=("z", "y", "x")),
        LinesMemoryStore(positions=np.zeros((2, 3)), axis_names=("z", "y", "x")),
        GraphMemoryStore(
            positions=np.zeros((2, 3)),
            edges=np.array([[0, 1]]),
            axis_names=("z", "y", "x"),
        ),
    ],
    ids=["points", "lines", "graph"],
)
def test_a_geometry_store_defaults_to_continuous(store):
    """A vertex coordinate is a measured position unless stated otherwise."""
    axes = store.data_coordinate_systems[0].axes
    assert [axis.sampling for axis in axes] == ["continuous"] * 3


def test_a_geometry_store_can_declare_one_axis_discrete():
    """The motivating case: t holds frame numbers, zyx hold positions.

    Per-axis rather than per-store, because a tracking graph is genuinely
    mixed.
    """
    store = GraphMemoryStore(
        positions=np.zeros((2, 4)),
        edges=np.array([[0, 1]]),
        axis_names=("t", "z", "y", "x"),
        axis_types=("time", "space", "space", "space"),
        axis_sampling=("discrete", "continuous", "continuous", "continuous"),
    )
    axes = store.data_coordinate_systems[0].axes
    assert [axis.sampling for axis in axes] == [
        "discrete",
        "continuous",
        "continuous",
        "continuous",
    ]


def test_axis_sampling_without_axis_names_raises():
    with pytest.raises(ValueError, match="need axis_names"):
        PointsMemoryStore(positions=np.zeros((2, 3)), axis_sampling=("discrete",))


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
