"""Tests for ImageAppearance and MultiscaleImageVisual models."""

import uuid

from cellier.v2.visuals._image import ImageAppearance, MultiscaleImageVisual


def test_image_appearance_roundtrip(tmp_path):
    # Non-default values including the new LOD/frustum fields
    original = ImageAppearance(
        color_map="viridis",
        clim=(0.1, 0.9),
        lod_bias=2.0,
        force_level=1,
        frustum_cull=False,
    )
    path = tmp_path / "appearance.json"
    path.write_text(original.model_dump_json())
    deserialized = ImageAppearance.model_validate_json(path.read_text())
    assert original.model_dump_json() == deserialized.model_dump_json()

    # force_level=None roundtrip
    original_none = ImageAppearance(
        color_map="gray",
        force_level=None,
    )
    path2 = tmp_path / "appearance_none.json"
    path2.write_text(original_none.model_dump_json())
    deserialized_none = ImageAppearance.model_validate_json(path2.read_text())
    assert original_none.model_dump_json() == deserialized_none.model_dump_json()


def test_multiscale_image_visual_roundtrip(tmp_path):
    store_id = str(uuid.uuid4())
    original = MultiscaleImageVisual(
        name="volume",
        data_store_id=store_id,
        downscale_factors=[1, 2, 4],
        appearance=ImageAppearance(
            color_map="viridis",
            clim=(0.0, 1.0),
            lod_bias=1.5,
            force_level=None,
            frustum_cull=True,
        ),
    )
    path = tmp_path / "visual.json"
    path.write_text(original.model_dump_json())
    deserialized = MultiscaleImageVisual.model_validate_json(path.read_text())
    assert original.model_dump_json() == deserialized.model_dump_json()
