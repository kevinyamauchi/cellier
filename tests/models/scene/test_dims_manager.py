"""Test the DimsManager class."""

import json

from pydantic_core import from_json

from cellier.models.scene import CoordinateSystem, DimsManager


def test_dims_manager(tmp_path):
    """Test serialization/deserialization of DimsManager."""
    coordinate_system = CoordinateSystem(name="default", axis_label=("x", "y", "z"))
    dims_manager = DimsManager(
        coordinate_system=coordinate_system, displayed_dimensions=(0, 1, 2)
    )

    output_path = tmp_path / "test.json"
    with open(output_path, "w") as f:
        # serialize the model
        json.dump(dims_manager.model_dump(), f)

    # deserialize
    with open(output_path, "rb") as f:
        deserialized_dims = DimsManager.model_validate(
            from_json(f.read(), allow_partial=False)
        )

    assert deserialized_dims == dims_manager
