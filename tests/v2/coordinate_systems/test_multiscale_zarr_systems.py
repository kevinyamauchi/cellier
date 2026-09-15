"""``MultiscaleZarrDataStore`` takes its axes from a caller's level-0 system.

The store reads no axis metadata, so its coordinate systems come from the
caller or, left empty, from the scene the store is added to.  Either way they
are checked against the pyramid it opens: one per level, one axis per
dimension.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorstore as ts

from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from tests._v2 import data_system

_SHAPES = {"s0": (8, 8, 8), "s1": (4, 4, 4)}
_GEOMETRY = {
    "level_scales": [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
    "level_translations": [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
}


@pytest.fixture
def pyramid_path(tmp_path) -> str:
    """A two-level zyx zarr v3 pyramid on disk."""
    for name, shape in _SHAPES.items():
        ts.open(
            {
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": str(tmp_path / name)},
            },
            create=True,
            dtype=ts.uint8,
            shape=shape,
        ).result()
    return str(tmp_path)


def _from_numbers(path: str, **kwargs) -> MultiscaleZarrDataStore:
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=path, scale_names=list(_SHAPES), **_GEOMETRY, **kwargs
    )


def test_without_a_system_the_store_waits_for_a_scene(pyramid_path: str) -> None:
    store = _from_numbers(pyramid_path)
    assert store.data_coordinate_systems == []
    assert store.level_transforms == []


def test_a_level_zero_system_becomes_one_per_level(pyramid_path: str) -> None:
    system = data_system(("z", "y", "x"), sampling="discrete")
    store = _from_numbers(pyramid_path, data_coordinate_system=system)

    assert store.id == system.datastore_id
    level_zero, level_one = store.data_coordinate_systems
    assert level_zero.id == system.id
    assert level_one.axis_names() == ("z", "y", "x")
    assert [axis.sampling for axis in level_one.axes] == ["discrete"] * 3
    assert level_one.datastore_id == store.id
    # Installed at construction, not deferred until the store joins a scene.
    assert len(store.level_transforms) == 2
    np.testing.assert_allclose(np.diag(store.level_transforms[1].matrix)[:3], 2.0)


def test_the_systems_survive_a_round_trip(pyramid_path: str) -> None:
    store = _from_numbers(
        pyramid_path, data_coordinate_system=data_system(("z", "y", "x"))
    )
    restored = MultiscaleZarrDataStore.model_validate_json(store.model_dump_json())
    assert [system.id for system in restored.data_coordinate_systems] == [
        system.id for system in store.data_coordinate_systems
    ]


def test_one_system_for_two_levels_is_refused_at_construction(
    pyramid_path: str,
) -> None:
    """It used to be accepted here and fail only once added to a scene."""
    with pytest.raises(ValueError, match="2 resolution level"):
        MultiscaleZarrDataStore(
            zarr_path=pyramid_path,
            scale_names=list(_SHAPES),
            **_GEOMETRY,
            data_coordinate_systems=[data_system(("z", "y", "x"))],
        )


def test_a_system_of_the_wrong_rank_is_refused(pyramid_path: str) -> None:
    with pytest.raises(ValueError, match="has 2 axes"):
        _from_numbers(pyramid_path, data_coordinate_system=data_system(("y", "x")))
