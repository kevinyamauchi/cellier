"""Tests for cellier.transform_v2._base (D9, D13, D14)."""

from uuid import uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from cellier.transform_v2 import (
    AffineTransform,
    Axis,
    BaseTransform,
    CoordinateSystem,
)
from tests.transform_v2._use_cases import uc1, uc3


def zyx(name="data"):
    return CoordinateSystem(
        name=name,
        axes=tuple(Axis(name=n, axis_type="space", unit="micrometer") for n in "zyx"),
    )


def test_ndim_is_read_through_and_stored_nowhere_d9():
    """D9: if you find yourself adding an ndim field, stop."""
    assert "ndim" not in AffineTransform.model_fields
    assert "input_ndim" not in AffineTransform.model_fields
    assert "output_ndim" not in AffineTransform.model_fields

    _, _, transform = uc3()
    assert transform.input_ndim == 4
    assert transform.output_ndim == 5
    assert transform.input_ndim == transform.transform.ndims.source
    assert transform.output_ndim == transform.transform.ndims.target


def test_validate_against_passes_on_agreement_d9():
    data, world, transform = uc1()
    transform.validate_against(data, world)


def test_validate_against_raises_on_a_wrong_input_id_d9():
    _, world, transform = uc1()
    with pytest.raises(ValueError, match="Input coordinate system id"):
        transform.validate_against(zyx(), world)


def test_validate_against_raises_on_a_wrong_output_id_d9():
    data, _, transform = uc1()
    with pytest.raises(ValueError, match="Output coordinate system id"):
        transform.validate_against(data, zyx())


def test_validate_against_raises_on_an_input_rank_mismatch_d9():
    data, world, transform = uc1()
    wrong_rank = CoordinateSystem(
        name=data.name,
        id=data.id,
        axes=(*data.axes, Axis(name="w", axis_type="space")),
    )
    with pytest.raises(ValueError, match="Input rank mismatch"):
        transform.validate_against(wrong_rank, world)


def test_validate_against_raises_on_an_output_rank_mismatch_d9():
    data, world, transform = uc1()
    wrong_rank = CoordinateSystem(
        name=world.name,
        id=world.id,
        axes=(*world.axes, Axis(name="W", axis_type="space")),
    )
    with pytest.raises(ValueError, match="Output rank mismatch"):
        transform.validate_against(data, wrong_rank)


def test_transform_is_frozen_q13():
    _, _, transform = uc1()
    with pytest.raises(ValidationError):
        transform.name = "renamed"


def test_transform_carries_an_id_and_an_optional_name_d14():
    data, world, transform = uc1()
    assert transform.name is None
    assert transform.id != uuid4()
    named = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={"z": "Z", "y": "Y", "x": "X"},
        name="data to world",
    )
    assert named.name == "data to world"
    # two transforms joining the same pair have different ids (D14)
    assert named.id != transform.id


def test_base_transform_is_abstract():
    assert BaseTransform.__abstractmethods__


def test_base_transform_declares_the_full_method_list():
    for method in (
        "map_coordinates",
        "imap_coordinates",
        "map_direction",
        "imap_direction",
        "map_normal",
        "imap_normal",
        "map_bounding_box",
        "imap_bounding_box",
        "map_plane",
        "imap_plane",
        "map_region",
        "imap_region",
        "inverse",
    ):
        assert hasattr(BaseTransform, method), method


def test_the_wrapped_transform_is_a_transformnd_object_q2():
    from transformnd.base import Transform

    _, _, transform = uc1()
    assert isinstance(transform.transform, Transform)


def test_spaced_is_not_used_anywhere_in_the_package():
    """Section 2.1: Spaced.invert() is wrong, so it is never reachable."""
    import cellier.transform_v2 as package

    for name in package.__all__:
        assert "Spaced" not in type(getattr(package, name)).__name__


def test_matrices_are_float64_d20():
    """v1 coerced to float32; narrowing is the renderer's job at upload."""
    _, _, transform = uc1()
    assert transform.matrix.dtype == np.float64
