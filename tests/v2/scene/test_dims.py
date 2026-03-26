"""Tests for CoordinateSystem, DimsManager, and AxisAlignedSelection models."""

import pytest

from cellier.v2._state import AxisAlignedSelectionState
from cellier.v2.scene.dims import (
    AxisAlignedSelection,
    CoordinateSystem,
    DimsManager,
)


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
        selection=AxisAlignedSelection(
            displayed_axes=(0, 1, 2),
            slice_indices={},
        ),
    )
    path = tmp_path / "dims_3d.json"
    path.write_text(original_3d.model_dump_json())
    deserialized_3d = DimsManager.model_validate_json(path.read_text())
    assert original_3d.model_dump_json() == deserialized_3d.model_dump_json()

    # 2D slice through 3D volume
    original_2d = DimsManager(
        coordinate_system=CoordinateSystem(name="world", axis_labels=("z", "y", "x")),
        selection=AxisAlignedSelection(
            displayed_axes=(1, 2),
            slice_indices={0: 32},
        ),
    )
    path2 = tmp_path / "dims_2d.json"
    path2.write_text(original_2d.model_dump_json())
    deserialized_2d = DimsManager.model_validate_json(path2.read_text())
    assert original_2d.model_dump_json() == deserialized_2d.model_dump_json()


def test_axis_aligned_selection_to_state():
    sel = AxisAlignedSelection(
        displayed_axes=(1, 2),
        slice_indices={0: 42},
    )
    state = sel.to_state()
    assert isinstance(state, AxisAlignedSelectionState)
    assert state.displayed_axes == (1, 2)
    assert state.slice_indices == {0: 42}


def test_to_index_selection_3d():
    """3D data, all axes displayed, empty slice_indices."""
    state = AxisAlignedSelectionState(
        displayed_axes=(0, 1, 2),
        slice_indices={},
    )
    result = state.to_index_selection(ndim=3)
    assert result == (slice(None), slice(None), slice(None))


def test_to_index_selection_2d_from_3d():
    """3D data, 2 axes displayed, one sliced."""
    state = AxisAlignedSelectionState(
        displayed_axes=(1, 2),
        slice_indices={0: 42},
    )
    result = state.to_index_selection(ndim=3)
    assert result == (42, slice(None), slice(None))


def test_to_index_selection_5d():
    """5D data, 3 displayed axes, 2 sliced."""
    state = AxisAlignedSelectionState(
        displayed_axes=(2, 3, 4),
        slice_indices={0: 5, 1: 1},
    )
    result = state.to_index_selection(ndim=5)
    assert result == (5, 1, slice(None), slice(None), slice(None))


def test_dims_manager_validates_axis_coverage():
    """Mismatched axes should raise ValidationError."""
    with pytest.raises(ValueError, match="Axis coverage mismatch"):
        DimsManager(
            coordinate_system=CoordinateSystem(
                name="world", axis_labels=("z", "y", "x")
            ),
            selection=AxisAlignedSelection(
                displayed_axes=(0, 1),
                slice_indices={},  # Missing axis 2
            ),
        )


def test_dims_manager_to_state():
    dims = DimsManager(
        coordinate_system=CoordinateSystem(
            name="world", axis_labels=("t", "c", "z", "y", "x")
        ),
        selection=AxisAlignedSelection(
            displayed_axes=(2, 3, 4),
            slice_indices={0: 5, 1: 1},
        ),
    )
    state = dims.to_state()
    assert state.axis_labels == ("t", "c", "z", "y", "x")
    assert state.selection.displayed_axes == (2, 3, 4)
    assert state.selection.slice_indices == {0: 5, 1: 1}
