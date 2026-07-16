"""Fan-out tests for the ``cellier.convenience.OrthoViewer`` add_* methods.

``test_ortho_viewer.py`` already covers ``add_image``'s fan-out. These cover the
remaining add methods, each of which registers one shared data store and one
visual per panel named ``f"{name}_{key}"``.
"""

from __future__ import annotations

from cellier.convenience import OrthoViewer
from cellier.visuals._channel_appearance import ChannelAppearance
from cellier.visuals._image import MultiscaleImageAppearance
from cellier.visuals._label_memory import InMemoryLabelsAppearance
from cellier.visuals._labels import MultiscaleLabelsAppearance
from cellier.visuals._lines_memory import LinesMemoryAppearance
from cellier.visuals._mesh_memory import MeshFlatAppearance
from cellier.visuals._points_memory import PointsMarkerAppearance

_PANELS = {"xy", "xz", "yz", "vol"}


def _assert_fanned_out(ortho: OrthoViewer, visuals: dict, name: str) -> None:
    assert set(visuals) == _PANELS
    assert {v.name for v in visuals.values()} == {f"{name}_{k}" for k in _PANELS}
    assert len(ortho.controller._model.data.stores) == 1
    for scene in ortho.scenes.values():
        assert len(scene.visuals) == 1


def test_add_labels_fans_out(labels_store):
    ortho = OrthoViewer(("z", "y", "x"))
    visuals = ortho.add_labels(
        labels_store, appearance=InMemoryLabelsAppearance(), name="lbl"
    )
    _assert_fanned_out(ortho, visuals, "lbl")


def test_add_mesh_fans_out(mesh_store):
    ortho = OrthoViewer(("z", "y", "x"))
    visuals = ortho.add_mesh(mesh_store, appearance=MeshFlatAppearance(), name="m")
    _assert_fanned_out(ortho, visuals, "m")


def test_add_points_fans_out(points_store):
    ortho = OrthoViewer(("z", "y", "x"))
    visuals = ortho.add_points(
        points_store, appearance=PointsMarkerAppearance(), name="pts"
    )
    _assert_fanned_out(ortho, visuals, "pts")


def test_add_lines_fans_out(lines_store):
    ortho = OrthoViewer(("z", "y", "x"))
    visuals = ortho.add_lines(
        lines_store, appearance=LinesMemoryAppearance(), name="ln"
    )
    _assert_fanned_out(ortho, visuals, "ln")


def test_add_image_multiscale_fans_out(multiscale_image_store):
    ortho = OrthoViewer(("z", "y", "x"))
    visuals = ortho.add_image_multiscale(
        multiscale_image_store,
        appearance=MultiscaleImageAppearance(color_map="viridis", render_mode="mip"),
        name="ms",
    )
    _assert_fanned_out(ortho, visuals, "ms")


def test_add_labels_multiscale_fans_out(multiscale_labels_store):
    ortho = OrthoViewer(("z", "y", "x"))
    visuals = ortho.add_labels_multiscale(
        multiscale_labels_store,
        appearance=MultiscaleLabelsAppearance(),
        name="mslbl",
    )
    _assert_fanned_out(ortho, visuals, "mslbl")


def test_add_multichannel_image_multiscale_fans_out(multichannel_multiscale_store):
    ortho = OrthoViewer(("c", "z", "y", "x"), spatial_axes=("z", "y", "x"))
    channels = {
        0: ChannelAppearance(color_map="red", clim=(0.0, 1.0)),
        1: ChannelAppearance(color_map="green", clim=(0.0, 1.0)),
    }
    visuals = ortho.add_multichannel_image_multiscale(
        multichannel_multiscale_store,
        channel_axis=0,
        channels=channels,
        name="mc",
    )
    _assert_fanned_out(ortho, visuals, "mc")
    for visual in visuals.values():
        assert set(visual.channels) == {0, 1}


def test_add_labels_accepts_dict_appearance(labels_store):
    ortho = OrthoViewer(("z", "y", "x"))
    visuals = ortho.add_labels(labels_store, appearance={})
    for visual in visuals.values():
        assert isinstance(visual.appearance, InMemoryLabelsAppearance)


def test_add_mesh_accepts_dict_appearance(mesh_store):
    ortho = OrthoViewer(("z", "y", "x"))
    visuals = ortho.add_mesh(mesh_store, appearance={"appearance_type": "flat"})
    for visual in visuals.values():
        assert isinstance(visual.appearance, MeshFlatAppearance)
