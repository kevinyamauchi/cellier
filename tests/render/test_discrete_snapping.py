"""Snapping a slice position on a discrete axis (snap_discrete_positions).

The unit-level counterpart to ``test_demo5_invariants.py``: that file checks
the graph and the image agree in a real scene, this one checks the rule they
agree by.
"""

from uuid import uuid4

import numpy as np
import pytest

from cellier.render._spaces import snap_discrete_positions
from cellier.transform import Axis, DataCoordinateSystem


class _Bounded:
    """A stand-in transform that reports a domain and nothing else."""

    def __init__(self, domain: dict[int, tuple[float, float]]) -> None:
        self._domain = domain

    def input_domain(self) -> dict[int, tuple[float, float]]:
        return self._domain


def _system(*sampling: str) -> DataCoordinateSystem:
    return DataCoordinateSystem(
        name="store",
        datastore_id=uuid4(),
        axes=tuple(
            Axis(name=f"a{i}", axis_type="space", sampling=value)
            for i, value in enumerate(sampling)
        ),
    )


def test_a_discrete_axis_snaps_to_the_nearest_sample():
    system = _system("discrete", "discrete")
    assert snap_discrete_positions({0: 6.4, 1: 2.1}, system) == {0: 6.0, 1: 2.0}


def test_a_continuous_axis_is_untouched():
    """A store whose column holds real measurements keeps containment."""
    system = _system("continuous")
    assert snap_discrete_positions({0: 6.4}, system) == {0: 6.4}


def test_axes_are_decided_independently():
    """A tracking graph is genuinely mixed: t indexed, zyx measured."""
    system = _system("discrete", "continuous", "continuous")
    snapped = snap_discrete_positions({0: 6.4, 1: 2.7, 2: 3.3}, system)
    assert snapped == {0: 6.0, 1: 2.7, 2: 3.3}


def test_ties_round_half_up():
    """The same rule round_world_to_voxel uses, which is the whole point."""
    system = _system("discrete")
    assert snap_discrete_positions({0: 1.5}, system) == {0: 2.0}
    assert snap_discrete_positions({0: 2.5}, system) == {0: 3.0}


def test_a_negative_position_still_rounds_half_up():
    system = _system("discrete")
    assert snap_discrete_positions({0: -0.5}, system) == {0: 0.0}
    assert snap_discrete_positions({0: -0.6}, system) == {0: -1.0}


def test_without_a_transform_it_does_not_confine():
    """With no domain to consult there is nothing to confine rounding to.

    An affine axis maps every real coordinate, so rounding cannot leave
    anything and no confinement is needed.
    """
    system = _system("discrete")
    assert snap_discrete_positions({0: 97.4}, system) == {0: 97.0}


def test_rounding_cannot_leave_the_samples_that_exist():
    """The crash reported from demo 5, at the very top of the slider.

    A 13-sample axis spans index ``(-0.5, 12.5)``.  The interval semantics
    of ``imap_bounding_box`` already clamp any slider position to that span,
    so the top of the slider arrives as **12.5** -- but round-half-up sends
    that boundary *up*, to a sample 13 that does not exist, and mapping it
    forward again returns nan.  Confining the result to the whole samples
    inside the domain is what keeps the last frame displayed.
    """
    system = _system("discrete")
    transform = _Bounded({0: (-0.5, 12.5)})
    assert snap_discrete_positions({0: 12.5}, system, transform) == {0: 12.0}


def test_the_bottom_edge_is_confined_too():
    """Symmetry, even though round-half-up only overshoots at the top."""
    system = _system("discrete")
    transform = _Bounded({0: (-0.5, 12.5)})
    assert snap_discrete_positions({0: -0.5}, system, transform) == {0: 0.0}


def test_an_axis_with_no_domain_is_left_unconfined():
    """An affine block reports no domain, so its axis is not confined."""
    system = _system("discrete", "discrete")
    transform = _Bounded({0: (-0.5, 12.5)})
    snapped = snap_discrete_positions({0: 12.5, 1: 97.4}, system, transform)
    assert snapped == {0: 12.0, 1: 97.0}


def test_confinement_is_not_out_of_range_clamping():
    """A position well inside the domain is untouched by the confinement."""
    system = _system("discrete")
    transform = _Bounded({0: (-0.5, 12.5)})
    assert snap_discrete_positions({0: 6.4}, system, transform) == {0: 6.0}


def test_a_non_finite_position_is_left_alone():
    """nan means the transform reported no preimage; the caller handles it."""
    system = _system("discrete")
    assert np.isnan(snap_discrete_positions({0: float("nan")}, system)[0])


def test_an_axis_outside_the_system_is_ignored():
    system = _system("discrete")
    assert snap_discrete_positions({5: 6.4}, system) == {5: 6.4}


def test_the_input_mapping_is_not_mutated():
    system = _system("discrete")
    positions = {0: 6.4}
    snap_discrete_positions(positions, system)
    assert positions == {0: 6.4}


@pytest.mark.parametrize(
    "raw",
    # The last three are the edges of the domain, where an earlier version of
    # this test stopped short -- it sampled only the interior and so passed
    # while the two rules disagreed at exactly 12.5.
    [-0.5, 0.0, 0.49, 0.5, 0.51, 6.4, 11.5, 12.0, 12.4, 12.5],
)
def test_snapping_agrees_with_round_world_to_voxel(raw):
    """The two must not drift: they are the same decision in two families.

    ``round_world_to_voxel`` confines its result to ``[0, size - 1]``; the
    geometry path confines its result to the samples inside the transform's
    domain.  Those have to be the same set, or an image and a graph over the
    same axis disagree about which frame is current.
    """
    from cellier.render.visuals._slicing import round_world_to_voxel

    system = _system("discrete")
    transform = _Bounded({0: (-0.5, 12.5)})
    snapped = snap_discrete_positions({0: raw}, system, transform)[0]
    assert snapped == float(round_world_to_voxel(raw, 13))
