"""Every store type round-trips through the ``DataStoreType`` union.

The union is what serializes a scene's stores.  A store class that is
exported and accepted by the controller but missing from the union cannot
survive a round trip, which is a silent hole rather than an error: the store
is written out fine and fails only on the way back in.
"""

from __future__ import annotations

import typing

import numpy as np
import pytest
from pydantic import TypeAdapter

import cellier.data as data_module
from cellier.data._base_data_store import BaseDataStore
from cellier.data._types import DataStoreType


def _union_members() -> set[type]:
    """The concrete classes ``DataStoreType`` discriminates between."""
    annotated_args = typing.get_args(DataStoreType)
    return set(typing.get_args(annotated_args[0]))


def _exported_stores() -> set[type]:
    """Every concrete ``BaseDataStore`` subclass exported from ``cellier.data``."""
    found = set()
    for name in data_module.__all__:
        obj = getattr(data_module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseDataStore)
            and obj is not BaseDataStore
        ):
            found.add(obj)
    return found


def test_every_exported_store_is_in_the_union():
    """``OMEZarrLabelDataStore`` was exported and usable but absent from the union.

    Nothing caught it because the omission only bites on deserialization: a
    scene containing one serialized cleanly and then failed to validate back.
    """
    missing = _exported_stores() - _union_members()
    assert not missing, (
        "exported store(s) missing from DataStoreType: "
        f"{sorted(cls.__name__ for cls in missing)}"
    )


def test_union_discriminators_are_unique():
    """Two stores sharing a ``store_type`` would silently resolve to one class."""
    discriminators = [
        cls.model_fields["store_type"].default for cls in _union_members()
    ]
    assert len(discriminators) == len(set(discriminators))


@pytest.mark.parametrize(
    "store",
    [
        data_module.ImageMemoryStore(data=np.zeros((2, 3, 4), dtype=np.float32)),
        data_module.LabelMemoryStore(data=np.zeros((2, 3, 4), dtype=np.int32)),
        data_module.PointsMemoryStore(positions=np.zeros((5, 3), dtype=np.float32)),
        data_module.LinesMemoryStore(positions=np.zeros((6, 3), dtype=np.float32)),
        data_module.MeshMemoryStore(
            positions=np.zeros((6, 3), dtype=np.float32),
            indices=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
        ),
        data_module.GraphMemoryStore(
            positions=np.zeros((4, 3), dtype=np.float32),
            edges=np.array([[0, 1]], dtype=np.int32),
        ),
    ],
    ids=lambda s: type(s).__name__,
)
def test_in_memory_stores_round_trip_through_the_union(store):
    """The discriminator resolves back to the class it was written from."""
    restored = TypeAdapter(DataStoreType).validate_python(store.model_dump())
    assert type(restored) is type(store)
    assert restored.store_type == store.store_type
