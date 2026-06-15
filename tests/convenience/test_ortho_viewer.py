"""Tests for cellier.convenience.OrthoViewer."""

import numpy as np
import pytest

from cellier.convenience import OrthoViewer, Viewer, axis_ranges_from_ortho
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.visuals._image_memory import InMemoryImageAppearance


@pytest.fixture
def image_store() -> ImageMemoryStore:
    data = np.zeros((8, 16, 24), dtype=np.float32)
    return ImageMemoryStore(data=data, name="test_image")


def test_scene_layout_3d():
    """The four panels get the expected displayed/sliced axes and render modes."""
    viewer = OrthoViewer(axis_labels=("z", "y", "x"))
    assert set(viewer.scenes) == {"xy", "xz", "yz", "vol"}
    assert viewer.spatial_axes == (0, 1, 2)
    assert viewer.extra_axes == set()

    expected_displayed = {
        "xy": (1, 2),
        "xz": (0, 2),
        "yz": (0, 1),
        "vol": (0, 1, 2),
    }
    for key, scene in viewer.scenes.items():
        assert tuple(scene.dims.selection.displayed_axes) == expected_displayed[key]
        assert scene.render_modes == ({"3d"} if key == "vol" else {"2d"})


def test_extra_axis_is_last_three_spatial_by_default():
    """A 5D dataset keeps the last three axes spatial; the rest are extra."""
    viewer = OrthoViewer(axis_labels=("t", "c", "z", "y", "x"))
    assert viewer.spatial_axes == (2, 3, 4)
    assert viewer.extra_axes == {0, 1}
    # Every panel slices both extra axes.
    for scene in viewer.scenes.values():
        sliced = set(scene.dims.selection.slice_indices)
        assert {0, 1} <= sliced


def test_explicit_spatial_axes_by_name():
    """spatial_axes may be given as names in (z, y, x) order."""
    viewer = OrthoViewer(axis_labels=("z", "y", "x", "c"), spatial_axes=("z", "y", "x"))
    assert viewer.spatial_axes == (0, 1, 2)
    assert viewer.extra_axes == {3}


def test_requires_three_axes():
    with pytest.raises(ValueError, match="at least 3 axes"):
        OrthoViewer(axis_labels=("y", "x"))


def test_add_image_fans_out_to_all_panels(image_store):
    """One add_image registers a single store and adds one visual per panel."""
    viewer = OrthoViewer(axis_labels=("z", "y", "x"))
    viewer.controller.add_data_store(image_store)
    visuals = viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(color_map="grays", clim=(0.0, 1.0)),
        name="blobs",
    )

    assert set(visuals) == {"xy", "xz", "yz", "vol"}
    assert {v.name for v in visuals.values()} == {
        "blobs_xy",
        "blobs_xz",
        "blobs_yz",
        "blobs_vol",
    }
    # Exactly one data store, one visual per scene.
    assert len(viewer.controller._model.data.stores) == 1
    for scene in viewer.scenes.values():
        assert len(scene.visuals) == 1


def test_center_slices_sets_integer_midpoints(image_store):
    viewer = OrthoViewer(axis_labels=("z", "y", "x"))
    viewer.controller.add_data_store(image_store)
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(color_map="grays", clim=(0.0, 1.0)),
    )
    ranges = axis_ranges_from_ortho(viewer)
    assert ranges == {0: (0.0, 7.0), 1: (0.0, 15.0), 2: (0.0, 23.0)}

    viewer.center_slices()
    assert dict(viewer.scenes["xy"].dims.selection.slice_indices) == {0: 4}
    assert dict(viewer.scenes["xz"].dims.selection.slice_indices) == {1: 8}
    assert dict(viewer.scenes["yz"].dims.selection.slice_indices) == {2: 12}
    # vol displays all spatial axes, so it has nothing to center.
    assert dict(viewer.scenes["vol"].dims.selection.slice_indices) == {}


def test_extra_axis_sync_propagates_across_panels():
    """Changing an extra axis on one panel updates the others."""
    viewer = OrthoViewer(axis_labels=("c", "z", "y", "x"))
    assert viewer.extra_axis_sync_enabled

    xy = viewer.scenes["xy"]
    new = dict(xy.dims.selection.slice_indices)
    new[0] = 3  # move the channel axis
    viewer.controller.update_slice_indices(xy.id, new)

    for scene in viewer.scenes.values():
        assert scene.dims.selection.slice_indices[0] == 3


def test_extra_axis_sync_can_be_disabled():
    viewer = OrthoViewer(axis_labels=("c", "z", "y", "x"))
    viewer.extra_axis_sync_enabled = False
    assert not viewer.extra_axis_sync_enabled

    xy = viewer.scenes["xy"]
    new = dict(xy.dims.selection.slice_indices)
    new[0] = 3
    viewer.controller.update_slice_indices(xy.id, new)

    # Other panels are untouched.
    assert viewer.scenes["vol"].dims.selection.slice_indices[0] == 0


def test_serialization_roundtrip(tmp_path, image_store):
    """OrthoViewer serializes and restores an equivalent ViewerModel."""
    viewer = OrthoViewer(axis_labels=("z", "y", "x"))
    viewer.controller.add_data_store(image_store)
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(color_map="grays", clim=(0.0, 1.0)),
        name="blobs",
    )
    viewer.center_slices()

    path = tmp_path / "ortho.json"
    viewer.to_file(path)

    loaded = OrthoViewer.from_file(path)
    assert viewer.controller._model == loaded.controller._model
    assert set(loaded.scenes) == {"xy", "xz", "yz", "vol"}
    assert loaded.spatial_axes == (0, 1, 2)


def test_from_file_rejects_non_ortho_model(tmp_path, image_store):
    """A single-scene Viewer file is not a valid OrthoViewer file."""
    viewer = Viewer(axis_labels=("z", "y", "x"))
    viewer.controller.add_data_store(image_store)
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(color_map="grays", clim=(0.0, 1.0)),
    )
    path = tmp_path / "single.json"
    viewer.to_file(path)

    with pytest.raises(ValueError, match="four orthoviewer panels"):
        OrthoViewer.from_file(path)
