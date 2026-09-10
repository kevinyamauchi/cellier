"""The OME-Zarr readers hand their axis metadata to a coordinate system.

NGFF already carries names, types and units, and cellier discarded all but
the names.  Populating ``data_coordinate_systems`` from them is close to free
and is a strict information gain -- and it is the one store family that can
answer D3's "what type is this axis" without asking the scene.
"""

from __future__ import annotations

import pathlib

import pytest

from cellier.data.image._ome_zarr_image_store import OMEZarrImageDataStore
from cellier.transform import AffineTransform as V1Affine
from cellier.transform_v2 import DataCoordinateSystem
from tests.v2.data.test_ome_zarr_image_store import _DATASETS


def test_the_level_zero_system_mirrors_the_ngff_axes(ome_zarr_5d: str) -> None:
    store = OMEZarrImageDataStore.from_path(ome_zarr_5d)
    system = store.data_coordinate_system
    assert isinstance(system, DataCoordinateSystem)
    assert system.axis_names() == ("t", "c", "z", "y", "x")
    assert [axis.axis_type for axis in system.axes] == [
        "time",
        "channel",
        "space",
        "space",
        "space",
    ]
    assert [axis.unit for axis in system.axes] == [
        "second",
        None,
        "micrometer",
        "micrometer",
        "micrometer",
    ]
    assert system.datastore_id == store.id


def test_there_is_one_system_per_pyramid_level(ome_zarr_5d: str) -> None:
    """A level-2 voxel is not a level-0 voxel, and a transform between them is
    exactly what says so -- which needs two distinct systems."""
    store = OMEZarrImageDataStore.from_path(ome_zarr_5d)
    assert len(store.data_coordinate_systems) == store.n_levels == 3
    ids = [system.id for system in store.data_coordinate_systems]
    assert len(set(ids)) == 3
    level_zero, level_one = store.data_coordinate_systems[:2]
    assert level_one.axis_names() == level_zero.axis_names()
    # Fresh axis ids per level: index_of must not be ambiguous across systems.
    assert {axis.id for axis in level_one.axes}.isdisjoint(
        {axis.id for axis in level_zero.axes}
    )


def test_the_systems_survive_a_round_trip_with_their_ids(ome_zarr_5d: str) -> None:
    store = OMEZarrImageDataStore.from_path(ome_zarr_5d)
    restored = OMEZarrImageDataStore.model_validate_json(store.model_dump_json())
    for mine, theirs in zip(
        store.data_coordinate_systems, restored.data_coordinate_systems
    ):
        assert theirs.id == mine.id
        assert [axis.id for axis in theirs.axes] == [axis.id for axis in mine.axes]


def test_an_empty_axis_type_raises_and_names_the_axis(ome_zarr_5d: str) -> None:
    """F1.4: the metadata had a slot for the type and left it blank, which is
    a dataset defect.  ``AxisType`` is a closed six-value literal with no
    honest fallback, so D3's "no silent default" is taken literally.

    Reached by constructing the store directly rather than through
    ``from_path``: the v0.5 image reader runs NGFF metadata through
    ``ome_zarr_models``, which rejects an untyped axis first (it can no longer
    order the axes).  The label reader parses raw dicts with
    ``ax.get("type", "")`` and has no such gate, so this is the shape a blank
    type actually arrives in.
    """
    store_path = ome_zarr_5d.removeprefix("file://")
    identity = V1Affine.identity(ndim=5)
    with pytest.raises(ValueError, match="'z' has an empty axis_type"):
        OMEZarrImageDataStore(
            zarr_path=ome_zarr_5d,
            scale_names=[dataset["path"] for dataset in _DATASETS],
            level_transforms=[identity, identity, identity],
            axis_names=["t", "c", "z", "y", "x"],
            axis_units=[None] * 5,
            axis_types=["time", "channel", "", "space", "space"],
        )
    assert pathlib.Path(store_path).exists()
