"""Tests for LabelMemoryAppearance and LabelMemoryVisual model layer."""

from __future__ import annotations

from cellier.v2.visuals._label_memory import LabelMemoryAppearance, LabelMemoryVisual

# ── LabelMemoryAppearance ────────────────────────────────────────────────────


def test_appearance_defaults():
    a = LabelMemoryAppearance()
    assert a.colormap_mode == "random"
    assert a.background_label == 0
    assert a.salt == 0
    assert a.color_dict == {}
    assert a.render_mode == "iso_categorical"
    assert a.opacity == 1.0


def test_appearance_json_roundtrip():
    a = LabelMemoryAppearance(
        colormap_mode="direct",
        background_label=-1,
        salt=42,
        color_dict={1: (1.0, 0.0, 0.0, 1.0), 2: (0.0, 1.0, 0.0, 1.0)},
        render_mode="flat_categorical",
        opacity=0.7,
    )
    json_str = a.model_dump_json()
    b = LabelMemoryAppearance.model_validate_json(json_str)
    assert b.colormap_mode == "direct"
    assert b.background_label == -1
    assert b.salt == 42
    assert b.color_dict == {1: (1.0, 0.0, 0.0, 1.0), 2: (0.0, 1.0, 0.0, 1.0)}
    assert b.render_mode == "flat_categorical"
    assert abs(b.opacity - 0.7) < 1e-6


def test_appearance_mutation_fires_event():
    a = LabelMemoryAppearance()
    received = []
    a.events.background_label.connect(lambda v: received.append(v))
    a.background_label = 5
    assert received == [5]


def test_appearance_salt_mutation_fires_event():
    a = LabelMemoryAppearance()
    received = []
    a.events.salt.connect(lambda v: received.append(v))
    a.salt = 123
    assert received == [123]


# ── LabelMemoryVisual ────────────────────────────────────────────────────────


def test_visual_json_roundtrip():
    from uuid import uuid4

    store_id = str(uuid4())
    a = LabelMemoryAppearance(
        colormap_mode="random",
        background_label=0,
        color_dict={7: (0.1, 0.2, 0.3, 1.0)},
    )
    v = LabelMemoryVisual(
        name="test",
        data_store_id=store_id,
        appearance=a,
    )
    json_str = v.model_dump_json()
    v2 = LabelMemoryVisual.model_validate_json(json_str)
    assert v2.data_store_id == store_id
    assert v2.appearance.colormap_mode == "random"
    assert v2.appearance.color_dict == {7: (0.1, 0.2, 0.3, 1.0)}


def test_visual_type_discriminator():
    from uuid import uuid4

    v = LabelMemoryVisual(
        name="x",
        data_store_id=str(uuid4()),
        appearance=LabelMemoryAppearance(),
    )
    assert v.visual_type == "label_memory"


def test_visual_requires_camera_reslice_is_false():
    from uuid import uuid4

    v = LabelMemoryVisual(
        name="x",
        data_store_id=str(uuid4()),
        appearance=LabelMemoryAppearance(),
    )
    assert v.requires_camera_reslice is False
