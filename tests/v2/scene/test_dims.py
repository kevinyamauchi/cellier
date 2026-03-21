"""Tests for CoordinateSystem and DimsManager models."""

from cellier.v2.scene.dims import CoordinateSystem, DimsManager


def test_coordinate_system_roundtrip(tmp_path):
    original = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    path = tmp_path / "coordinate_system.json"
    path.write_text(original.model_dump_json())
    deserialized = CoordinateSystem.model_validate_json(path.read_text())
    assert original.model_dump_json() == deserialized.model_dump_json()


def test_dims_manager_roundtrip(tmp_path):
    # 3D case — all axes displayed, no slice indices
    original_3d = DimsManager(
        coordinate_system=CoordinateSystem(name="world", axis_labels=("z", "y", "x")),
        displayed_axes=(0, 1, 2),
        slice_indices=(),
    )
    path = tmp_path / "dims_3d.json"
    path.write_text(original_3d.model_dump_json())
    deserialized_3d = DimsManager.model_validate_json(path.read_text())
    assert original_3d.model_dump_json() == deserialized_3d.model_dump_json()

    # 2D slice through 3D volume
    original_2d = DimsManager(
        coordinate_system=CoordinateSystem(name="world", axis_labels=("z", "y", "x")),
        displayed_axes=(1, 2),
        slice_indices=(32,),
    )
    path2 = tmp_path / "dims_2d.json"
    path2.write_text(original_2d.model_dump_json())
    deserialized_2d = DimsManager.model_validate_json(path2.read_text())
    assert original_2d.model_dump_json() == deserialized_2d.model_dump_json()
