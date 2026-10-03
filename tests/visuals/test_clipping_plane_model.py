"""``ClippingPlane`` and the ``clipping_planes`` field of a visual."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from cellier.transform import Axis, CoordinateSystem, Plane
from cellier.visuals import ClippingPlane, PointsMarkerAppearance, PointsVisual

SYSTEM = CoordinateSystem(
    name="data", axes=tuple(Axis(name=n, axis_type="space") for n in "tzyx")
)


def _plane(position=40.0, enabled=True) -> ClippingPlane:
    return ClippingPlane.from_point_normal(
        SYSTEM, (position, 0, 0), (1, 0, 0), axes=("z", "y", "x"), enabled=enabled
    )


def _visual() -> PointsVisual:
    return PointsVisual(
        name="points", data_store_id=str(uuid4()), appearance=PointsMarkerAppearance()
    )


def test_from_point_normal_forwards_to_the_plane():
    plane = _plane()
    assert plane.enabled is True
    assert plane.plane == Plane.from_point_normal(
        SYSTEM, (40, 0, 0), (1, 0, 0), axes=("z", "y", "x")
    )


def test_a_clipping_plane_is_frozen():
    with pytest.raises(ValidationError):
        _plane().enabled = False


def test_one_event_per_assignment_that_changes_the_value():
    visual = _visual()
    seen: list = []
    visual.events.clipping_planes.connect(seen.append)

    visual.clipping_planes = (_plane(),)
    assert len(seen) == 1
    visual.clipping_planes = (_plane(),)  # equal, built from scratch
    assert len(seen) == 1
    visual.clipping_planes = (_plane(41.0),)
    assert len(seen) == 2
    visual.clipping_planes = (_plane(41.0, enabled=False),)
    assert len(seen) == 3
    visual.clipping_planes = ()
    assert len(seen) == 4
    assert seen[-1] == ()


def test_assignment_is_validated():
    visual = _visual()
    visual.clipping_planes = [_plane()]
    assert isinstance(visual.clipping_planes, tuple)
    with pytest.raises(ValidationError):
        visual.clipping_planes = ("not a plane",)
    assert visual.clipping_planes == (_plane(),)


def test_equality_is_total():
    a, b = _plane(), _plane(41.0)
    assert (a == _plane()) is True
    assert (a == b) is False
    assert (a == _plane(enabled=False)) is False
    assert hash(a) == hash(_plane())
    short = ClippingPlane(
        plane=Plane(coordinate_system=SYSTEM.id, normal=(1, 0), offset=1)
    )
    assert (a == short) is False


def test_json_round_trip_is_exact():
    visual = _visual()
    visual.clipping_planes = (_plane(40.25), _plane(3.0, enabled=False))
    restored = PointsVisual.model_validate_json(visual.model_dump_json())
    assert restored.clipping_planes == visual.clipping_planes
