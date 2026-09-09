"""Tests for cellier.transform_v2._coordinate_system (D6-D8, D33, R5)."""

from functools import partial
from uuid import uuid4

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from cellier.transform_v2 import (
    Axis,
    CoordinateSystem,
    CoordinateSystemType,
    DataCoordinateSystem,
    RenderedCoordinateSystem,
    WorldCoordinateSystem,
)

space = partial(Axis, axis_type="space", unit="micrometer")


def zyx() -> CoordinateSystem:
    return CoordinateSystem(
        name="data", axes=(space(name="z"), space(name="y"), space(name="x"))
    )


def tczyx_world() -> WorldCoordinateSystem:
    return WorldCoordinateSystem(
        axes=(
            Axis(name="T", axis_type="time", unit="second"),
            Axis(name="C", axis_type="channel"),
            space(name="Z"),
            space(name="Y"),
            space(name="X"),
        )
    )


# --- CoordinateSystem ------------------------------------------------


def test_ndim_equals_len_axes():
    assert zyx().ndim == 3


def test_empty_axes_rejected_d6():
    with pytest.raises(ValidationError):
        CoordinateSystem(name="empty", axes=())


def test_duplicate_axis_ids_rejected_d6():
    axis = space(name="z")
    with pytest.raises(ValidationError):
        CoordinateSystem(name="dupe", axes=(axis, axis))


def test_duplicate_axis_names_accepted_q6():
    system = CoordinateSystem(name="dupe", axes=(space(name="z"), space(name="z")))
    assert system.axis_names() == ("z", "z")


def test_axis_by_name_unique_match():
    system = zyx()
    assert system.axis_by_name("y").name == "y"


def test_axis_by_name_raises_on_ambiguity_d6():
    system = CoordinateSystem(name="dupe", axes=(space(name="z"), space(name="z")))
    with pytest.raises(ValueError, match="ambiguous"):
        system.axis_by_name("z")


def test_axis_by_name_raises_on_miss():
    with pytest.raises(KeyError):
        zyx().axis_by_name("q")


def test_index_of_round_trips_against_axes():
    system = zyx()
    for index, axis in enumerate(system.axes):
        assert system.index_of(axis.id) == index


def test_index_of_raises_for_a_foreign_axis():
    with pytest.raises(KeyError):
        zyx().index_of(uuid4())


def test_resolve_accepts_a_name_or_an_id():
    system = zyx()
    assert system.resolve("y") == 1
    assert system.resolve(system.axes[1].id) == 1
    assert system.resolve_axis("y") is system.axes[1]


def test_resolve_by_id_works_where_a_name_is_ambiguous_d6():
    """An id is never ambiguous, even when the names collide."""
    first, second = space(name="z"), space(name="z")
    system = CoordinateSystem(name="dupe", axes=(first, second))
    with pytest.raises(ValueError, match="ambiguous"):
        system.resolve("z")
    assert system.resolve(second.id) == 1


def test_coordinate_system_is_frozen_d7():
    system = zyx()
    with pytest.raises(ValidationError):
        system.name = "other"


# --- subclasses ------------------------------------------------------


def test_data_coordinate_system_requires_datastore_id_q7():
    with pytest.raises(ValidationError):
        DataCoordinateSystem(name="cells", axes=(space(name="z"),))
    system = DataCoordinateSystem(
        name="cells", datastore_id=uuid4(), axes=(space(name="z"),)
    )
    assert system.coordinate_system_type == "data"


def test_world_coordinate_system_name_defaults_to_world():
    assert WorldCoordinateSystem(axes=(space(name="Z"),)).name == "world"


def test_discriminated_union_round_trips_over_a_heterogeneous_list_d8():
    class Holder(BaseModel):
        systems: tuple[CoordinateSystemType, ...]

    world = tczyx_world()
    holder = Holder(
        systems=(
            zyx(),
            DataCoordinateSystem(
                name="cells", datastore_id=uuid4(), axes=(space(name="z"),)
            ),
            world,
            RenderedCoordinateSystem.from_world(world, ("Z", "Y", "X"), uuid4()),
        )
    )
    back = Holder.model_validate(holder.model_dump())
    assert [type(s) for s in back.systems] == [
        CoordinateSystem,
        DataCoordinateSystem,
        WorldCoordinateSystem,
        RenderedCoordinateSystem,
    ]
    assert back == holder


def test_union_adapter_selects_by_discriminator():
    adapter = TypeAdapter(CoordinateSystemType)
    world = tczyx_world()
    parsed = adapter.validate_python(world.model_dump())
    assert isinstance(parsed, WorldCoordinateSystem)


# --- RenderedCoordinateSystem (section 7.1) --------------------------


def test_rendered_rank_other_than_2_or_3_is_rejected_d33():
    canvas_id = uuid4()
    with pytest.raises(ValidationError):
        RenderedCoordinateSystem(axes=(space(name="Z"),), canvas_id=canvas_id)
    with pytest.raises(ValidationError):
        RenderedCoordinateSystem.from_world(
            tczyx_world(), ("T", "C", "Z", "Y"), canvas_id
        )
    assert (
        RenderedCoordinateSystem.from_world(tczyx_world(), ("Z", "Y"), canvas_id).ndim
        == 2
    )


def test_from_world_inherits_metadata_and_makes_fresh_ids_r5():
    world = tczyx_world()
    rendered = RenderedCoordinateSystem.from_world(world, ("T", "Z", "Y"), uuid4())

    assert rendered.axis_names() == ("T", "Z", "Y")
    assert [a.axis_type for a in rendered.axes] == ["time", "space", "space"]
    assert [a.unit for a in rendered.axes] == ["second", "micrometer", "micrometer"]

    world_ids = {a.id for a in world.axes}
    assert not world_ids & {a.id for a in rendered.axes}


def test_displayed_axes_order_is_the_rendered_axis_order():
    world = tczyx_world()
    canvas_id = uuid4()
    xyz = RenderedCoordinateSystem.from_world(world, ("X", "Y", "Z"), canvas_id)
    zyx_ = RenderedCoordinateSystem.from_world(world, ("Z", "Y", "X"), canvas_id)
    assert xyz.axis_names() == ("X", "Y", "Z")
    assert zyx_.axis_names() == ("Z", "Y", "X")
    assert xyz.axis_names() != zyx_.axis_names()


def test_from_world_accepts_axis_ids():
    world = tczyx_world()
    rendered = RenderedCoordinateSystem.from_world(
        world, (world.axes[2].id, world.axes[3].id), uuid4()
    )
    assert rendered.axis_names() == ("Z", "Y")


def test_from_world_rejects_a_repeated_world_axis():
    with pytest.raises(ValueError, match="distinct"):
        RenderedCoordinateSystem.from_world(tczyx_world(), ("Z", "Z"), uuid4())


def test_rendered_has_no_world_backref_d33():
    """D33: the link is carried by the transform, not by a second field."""
    rendered = RenderedCoordinateSystem.from_world(tczyx_world(), ("Z", "Y"), uuid4())
    assert "world_coordinate_system" not in type(rendered).model_fields
