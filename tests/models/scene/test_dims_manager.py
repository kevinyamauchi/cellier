"""Test the DimsManager class."""

from pydantic_core import from_json

from cellier.models.scene import (
    AxisAlignedRegionSelector,
    CoordinateSystem,
    DimsManager,
    RangeTuple,
)
from cellier.types import CoordinateSpace


def test_axis_aligned_region_selector_serialization(tmp_path):
    """Test serialization/deserialization of AxisAlignedRegionSelector."""
    region_selector = AxisAlignedRegionSelector(
        space_type=CoordinateSpace.WORLD,
        ordered_dims=(0, 1, 2),
        n_displayed_dims=2,
        index_selection=(0, slice(0, 10, 1), slice(None, None, None)),
    )

    output_path = tmp_path / "test.json"
    with open(output_path, "w") as f:
        # serialize the model
        f.write(region_selector.model_dump_json())

    # deserialize
    with open(output_path, "rb") as f:
        deserialized_region_selector = AxisAlignedRegionSelector.model_validate(
            from_json(f.read(), allow_partial=False)
        )

    assert deserialized_region_selector == region_selector


def test_dims_manager_serialization(tmp_path):
    """Test serialization/deserialization of DimsManager."""
    coordinate_system = CoordinateSystem(name="default", axis_label=("x", "y", "z"))
    dims_manager = DimsManager(
        coordinate_system=coordinate_system,
        range=(RangeTuple(0, 10, 1), RangeTuple(0, 10, 1), RangeTuple(0, 10, 1)),
        selection=AxisAlignedRegionSelector(
            space_type=CoordinateSpace.WORLD,
            ordered_dims=(0, 1, 2),
            n_displayed_dims=2,
            index_selection=(0, slice(0, 10, 1), slice(None, None, None)),
        ),
    )

    output_path = tmp_path / "test.json"
    with open(output_path, "w") as f:
        # serialize the model
        f.write(dims_manager.model_dump_json())

    # deserialize
    with open(output_path, "rb") as f:
        deserialized_dims = DimsManager.model_validate(
            from_json(f.read(), allow_partial=False)
        )

    assert deserialized_dims == dims_manager
