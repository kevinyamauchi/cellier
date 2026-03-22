"""Unit tests for CellierController.

All tests are headless — no Qt event loop, no GPU.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from cellier.v2.controller import CellierController
from cellier.v2.data.image import MultiscaleZarrDataStore
from cellier.v2.scene.dims import CoordinateSystem
from cellier.v2.viewer_model import DataManager, ViewerModel
from cellier.v2.visuals._image import ImageAppearance, MultiscaleImageVisual

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cs() -> CoordinateSystem:
    return CoordinateSystem(name="world", axis_labels=("z", "y", "x"))


def _make_appearance(**kwargs) -> ImageAppearance:
    defaults = {"color_map": "viridis", "clim": (0.0, 1.0)}
    defaults.update(kwargs)
    return ImageAppearance(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_add_scene_registers_in_model():
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    assert scene.id in controller._model.scenes
    assert controller._model.scenes[scene.id].name == "main"


def test_add_data_store_registers_in_model(small_zarr_store):
    controller = CellierController()
    store = MultiscaleZarrDataStore(
        zarr_path=str(small_zarr_store), scale_names=["s0", "s1"]
    )
    result = controller.add_data_store(store)
    assert result is store
    assert store.id in controller._model.data.stores


def test_add_image_populates_scene_and_render_layer(small_zarr_store):
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = MultiscaleZarrDataStore(
        zarr_path=str(small_zarr_store), scale_names=["s0", "s1"]
    )
    appearance = _make_appearance(lod_bias=1.0, force_level=None, frustum_cull=True)
    visual = controller.add_image(
        data=store, scene_id=scene.id, appearance=appearance, name="vol"
    )

    assert isinstance(visual, MultiscaleImageVisual)
    # Model layer
    assert visual in controller._model.scenes[scene.id].visuals
    assert store.id in controller._model.data.stores
    # Render layer
    assert visual.id in controller._render_manager._visual_to_scene


def test_add_image_auto_registers_data_store(small_zarr_store):
    """add_image should register the data store even without add_data_store."""
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = MultiscaleZarrDataStore(
        zarr_path=str(small_zarr_store), scale_names=["s0", "s1"]
    )
    controller.add_image(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )
    assert store.id in controller._model.data.stores


def test_get_scene_by_name():
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="my_scene")
    found = controller.get_scene_by_name("my_scene")
    assert found.id == scene.id
    with pytest.raises(KeyError):
        controller.get_scene_by_name("nonexistent")


def test_reslice_scene_reads_appearance_fields(small_zarr_store):
    """reslice_scene must translate ImageAppearance fields into VisualRenderConfig."""
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = MultiscaleZarrDataStore(
        zarr_path=str(small_zarr_store), scale_names=["s0", "s1"]
    )
    appearance = _make_appearance(lod_bias=2.5, force_level=1, frustum_cull=False)
    visual = controller.add_image(
        data=store, scene_id=scene.id, appearance=appearance, name="vol"
    )

    captured = {}

    def capturing_reslice(s_id, dims_state, visual_configs=None):
        captured.update(visual_configs or {})

    controller._render_manager.reslice_scene = capturing_reslice

    controller.reslice_scene(scene.id)

    assert visual.id in captured
    cfg = captured[visual.id]
    assert cfg.lod_bias == 2.5
    assert cfg.force_level == 1
    assert cfg.frustum_cull is False


def test_to_file_roundtrip(tmp_path, small_zarr_store):
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = MultiscaleZarrDataStore(
        zarr_path=str(small_zarr_store), scale_names=["s0", "s1"]
    )
    controller.add_image(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )

    path = tmp_path / "session.json"
    controller.to_file(path)

    restored = ViewerModel.from_file(path)
    assert len(restored.scenes) == 1
    assert len(next(iter(restored.scenes.values())).visuals) == 1


def test_stubs_raise():
    controller = CellierController()
    with pytest.raises(NotImplementedError):
        controller.from_model(ViewerModel(data=DataManager()))
    with pytest.raises(NotImplementedError):
        controller.remove_scene(uuid4())
    with pytest.raises(NotImplementedError):
        controller.get_dims_widget(uuid4())


# ---------------------------------------------------------------------------
# Event bus tests (headless — no Qt, no GPU)
# ---------------------------------------------------------------------------


def test_dims_bridge_emits_event():
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")

    fired = []
    controller.on_dims_changed(scene.id, fired.append)

    controller.get_scene(scene.id).dims.slice_indices = (5,)
    assert len(fired) == 1
    assert fired[0].slice_indices == (5,)


def test_dims_bridge_displayed_axes_flag_true():
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")

    events = []
    controller._event_bus.subscribe(
        __import__("cellier.v2.events", fromlist=["DimsChangedEvent"]).DimsChangedEvent,
        events.append,
        entity_id=scene.id,
    )

    controller.get_scene(scene.id).dims.displayed_axes = (0, 1)
    assert len(events) == 1
    assert events[0].displayed_axes_changed is True


def test_dims_bridge_displayed_axes_flag_false():
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")

    events = []
    controller._event_bus.subscribe(
        __import__("cellier.v2.events", fromlist=["DimsChangedEvent"]).DimsChangedEvent,
        events.append,
        entity_id=scene.id,
    )

    controller.get_scene(scene.id).dims.slice_indices = (3,)
    assert len(events) == 1
    assert events[0].displayed_axes_changed is False


def test_appearance_bridge_color_map(small_zarr_store):
    from cellier.v2.events import AppearanceChangedEvent

    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = MultiscaleZarrDataStore(
        zarr_path=str(small_zarr_store), scale_names=["s0", "s1"]
    )
    visual = controller.add_image(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )

    events = []
    controller._event_bus.subscribe(
        AppearanceChangedEvent, events.append, entity_id=visual.id
    )

    visual.appearance.color_map = "plasma"
    assert len(events) == 1
    assert events[0].field_name == "color_map"
    assert events[0].requires_reslice is False


def test_appearance_bridge_lod_bias(small_zarr_store):
    from cellier.v2.events import AppearanceChangedEvent

    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = MultiscaleZarrDataStore(
        zarr_path=str(small_zarr_store), scale_names=["s0", "s1"]
    )
    visual = controller.add_image(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )

    events = []
    controller._event_bus.subscribe(
        AppearanceChangedEvent, events.append, entity_id=visual.id
    )

    visual.appearance.lod_bias = 2.0
    assert len(events) == 1
    assert events[0].requires_reslice is True


def test_appearance_bridge_force_level(small_zarr_store):
    from cellier.v2.events import AppearanceChangedEvent

    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = MultiscaleZarrDataStore(
        zarr_path=str(small_zarr_store), scale_names=["s0", "s1"]
    )
    visual = controller.add_image(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )

    events = []
    controller._event_bus.subscribe(
        AppearanceChangedEvent, events.append, entity_id=visual.id
    )

    visual.appearance.force_level = 1
    assert len(events) == 1
    assert events[0].requires_reslice is True


def test_appearance_bridge_visible(small_zarr_store):
    from cellier.v2.events import AppearanceChangedEvent, VisualVisibilityChangedEvent

    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = MultiscaleZarrDataStore(
        zarr_path=str(small_zarr_store), scale_names=["s0", "s1"]
    )
    visual = controller.add_image(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )

    appearance_events = []
    visibility_events = []
    controller._event_bus.subscribe(
        AppearanceChangedEvent, appearance_events.append, entity_id=visual.id
    )
    controller._event_bus.subscribe(
        VisualVisibilityChangedEvent, visibility_events.append, entity_id=visual.id
    )

    visual.appearance.visible = False
    assert len(visibility_events) == 1
    assert visibility_events[0].visible is False
    assert len(appearance_events) == 0


def test_scene_added_event_emitted():
    from cellier.v2.events import SceneAddedEvent

    controller = CellierController()
    events = []
    controller._event_bus.subscribe(SceneAddedEvent, events.append)

    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")

    assert len(events) == 1
    assert events[0].scene_id == scene.id


def test_visual_added_event_emitted(small_zarr_store):
    from cellier.v2.events import VisualAddedEvent

    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = MultiscaleZarrDataStore(
        zarr_path=str(small_zarr_store), scale_names=["s0", "s1"]
    )

    events = []
    controller._event_bus.subscribe(VisualAddedEvent, events.append)

    visual = controller.add_image(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )

    assert len(events) == 1
    assert events[0].visual_id == visual.id
    assert events[0].scene_id == scene.id


def test_on_dims_changed_callback():
    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")

    fired = []
    controller.on_dims_changed(scene.id, fired.append)

    controller.get_scene(scene.id).dims.slice_indices = (7,)
    assert len(fired) == 1
    assert fired[0].slice_indices == (7,)


def test_unsubscribe_all_cleans_up(small_zarr_store):
    from cellier.v2.events import AppearanceChangedEvent

    controller = CellierController()
    cs = _make_cs()
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = MultiscaleZarrDataStore(
        zarr_path=str(small_zarr_store), scale_names=["s0", "s1"]
    )
    visual = controller.add_image(
        data=store, scene_id=scene.id, appearance=_make_appearance(), name="vol"
    )

    fired = []
    controller._event_bus.subscribe(
        AppearanceChangedEvent,
        fired.append,
        entity_id=visual.id,
        owner_id=visual.id,
    )
    controller._event_bus.unsubscribe_all(visual.id)

    visual.appearance.color_map = "plasma"
    assert fired == []
