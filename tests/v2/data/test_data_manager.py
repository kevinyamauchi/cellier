"""Tests for DataManager model."""

from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.viewer_model import DataManager
from tests._v2 import pyramid_levels


def test_data_manager_roundtrip(tmp_path, small_zarr_store):
    store = MultiscaleZarrDataStore(
        zarr_path=str(small_zarr_store),
        scale_names=["s0", "s1"],
        **pyramid_levels(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        ),
        name="test_volume",
    )
    original = DataManager(stores={store.id: store})

    path = tmp_path / "data_manager.json"
    path.write_text(original.model_dump_json())
    deserialized = DataManager.model_validate_json(path.read_text())

    assert original.model_dump_json() == deserialized.model_dump_json()

    # Verify model_post_init ran on the deserialized store
    deserialized_store = next(iter(deserialized.stores.values()))
    assert deserialized_store.n_levels == 2
