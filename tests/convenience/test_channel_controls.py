"""Convenience-layer tests for the ChannelControls dock (Qt default path)."""

from __future__ import annotations

import numpy as np
import pytest

from cellier.convenience import ChannelControls, OrthoViewer, Viewer
from cellier.convenience._hosts import QtLayoutHost
from cellier.convenience.gui._controls_config import ChannelControlsConfig
from cellier.convenience.layout._shared import channel_targets
from cellier.convenience.layout._walk import render_dock
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.visuals._channel_appearance import ChannelAppearance

_AXES = [("z", "space"), ("c", "channel"), ("y", "space"), ("x", "space")]


def _make_channel_appearance(**kwargs) -> ChannelAppearance:
    defaults = {"color_map": "viridis", "clim": (0.0, 1.0)}
    defaults.update(kwargs)
    return ChannelAppearance(**defaults)


def _make_store(n_channels=2) -> ImageMemoryStore:
    data = np.random.default_rng(0).random((3, n_channels, 16, 16)).astype(np.float32)
    return ImageMemoryStore(data=data)


def _channels(n=2) -> dict[int, ChannelAppearance]:
    return {i: _make_channel_appearance() for i in range(n)}


def _find_control(widget, channel_index: int, field: str):
    applier = widget._appliers[(channel_index, field)]
    return applier.__closure__[0].cell_contents


# ---------------------------------------------------------------------------
# Resolver + cap validation (design section 7.3 / 11.4)
# ---------------------------------------------------------------------------


def test_resolver_single_viewer_returns_config_and_visual():
    viewer = Viewer(_AXES, dim="2d")
    visual = viewer.add_multichannel_image(
        _make_store(),
        channel_axis=1,
        channels=_channels(2),
        controls=ChannelControlsConfig(),
    )

    (target,) = channel_targets(viewer)
    assert isinstance(target.config, ChannelControlsConfig)
    assert target.visual is visual
    assert target.visual_ids == [visual.id]
    assert set(target.visual.channels) == {0, 1}


def test_resolver_returns_nothing_without_controls():
    viewer = Viewer(_AXES, dim="2d")
    viewer.add_multichannel_image(_make_store(), channel_axis=1, channels=_channels(2))
    assert channel_targets(viewer) == []


def test_resolver_returns_every_channel_visual_in_add_order():
    viewer = Viewer(_AXES, dim="2d")
    first = viewer.add_multichannel_image(
        _make_store(),
        channel_axis=1,
        channels=_channels(2),
        name="first",
        controls=ChannelControlsConfig(),
    )
    second = viewer.add_multichannel_image(
        _make_store(),
        channel_axis=1,
        channels=_channels(2),
        name="second",
        controls=ChannelControlsConfig(),
    )

    targets = channel_targets(viewer)
    assert [t.visual for t in targets] == [first, second]
    assert [t.label for t in targets] == ["first", "second"]


def test_over_cap_channel_controls_raise_at_the_add():
    """Raised where the mistake is made, not when a dock builds the widget."""
    viewer = Viewer(_AXES, dim="2d")
    with pytest.raises(ValueError, match="min\\(max_channels_2d"):
        viewer.add_multichannel_image(
            _make_store(3),
            channel_axis=1,
            channels=_channels(3),
            max_channels_3d=2,  # min(8, 2) = 2 < 3 channels
            controls=ChannelControlsConfig(),
        )
    # Refused before the controller saw it: nothing half-added.
    assert list(viewer.scene.visuals) == []
    assert viewer._controls_configs == {}


def test_over_cap_without_controls_is_not_checked():
    """The cap is a channel-controls constraint, not an add constraint."""
    viewer = Viewer(_AXES, dim="2d")
    viewer.add_multichannel_image(
        _make_store(3), channel_axis=1, channels=_channels(3), max_channels_3d=2
    )
    assert len(viewer.scene.visuals) == 1


def test_over_cap_channel_controls_raise_at_the_ortho_add():
    ortho = OrthoViewer(_AXES, spatial_axes=("z", "y", "x"))
    with pytest.raises(ValueError, match="min\\(max_channels_2d"):
        ortho.add_multichannel_image(
            _make_store(3),
            channel_axis=1,
            channels=_channels(3),
            max_channels_3d=2,
            controls=ChannelControlsConfig(),
        )
    assert all(list(scene.visuals) == [] for scene in ortho.scenes.values())


def test_resolver_succeeds_at_exactly_cap():
    viewer = Viewer(_AXES, dim="2d")
    viewer.add_multichannel_image(
        _make_store(3),
        channel_axis=1,
        channels=_channels(3),
        max_channels_3d=3,  # min(8, 3) = 3 == 3 channels
        controls=ChannelControlsConfig(),
    )
    (target,) = channel_targets(viewer)
    assert len(target.visual.channels) == 3


# ---------------------------------------------------------------------------
# OrthoViewer: one widget drives every panel
# ---------------------------------------------------------------------------


def test_ortho_resolver_gathers_all_panel_ids():
    ortho = OrthoViewer(_AXES, spatial_axes=("z", "y", "x"))
    visuals = ortho.add_multichannel_image(
        _make_store(2),
        channel_axis=1,
        channels=_channels(2),
        name="cells",
        controls=ChannelControlsConfig(),
    )

    (target,) = channel_targets(ortho)
    assert set(target.visual_ids) == {v.id for v in visuals.values()}
    assert len(target.visual_ids) == 4
    # Named by the add, not by the representative panel's "cells_xy".
    assert target.label == "cells"


def test_ortho_edit_reaches_all_panels(qtbot):
    from cellier.gui.qt.visuals import QtChannelList

    ortho = OrthoViewer(_AXES, spatial_axes=("z", "y", "x"))
    visuals = ortho.add_multichannel_image(
        _make_store(2),
        channel_axis=1,
        channels=_channels(2),
        controls=ChannelControlsConfig(),
    )

    (target,) = channel_targets(ortho)
    widget = QtChannelList(target.visual_ids, target.visual.channels)
    ortho.controller.connect_widget(
        widget, subscription_specs=widget.subscription_specs()
    )

    checkbox = _find_control(widget, 1, "visible")
    checkbox.setChecked(False)

    for visual in visuals.values():
        assert visual.channels[1].visible is False


# ---------------------------------------------------------------------------
# Qt renderer branch
# ---------------------------------------------------------------------------


def test_render_dock_qt_dispatches_channel_controls(qtbot):
    from cellier.gui.qt.visuals import QtChannelList

    viewer = Viewer(_AXES, dim="2d")
    viewer.add_multichannel_image(
        _make_store(2),
        channel_axis=1,
        channels=_channels(2),
        controls=ChannelControlsConfig(),
    )

    closeables: list = []
    rendered = render_dock(ChannelControls(), viewer, QtLayoutHost(), closeables)
    assert rendered is not None
    (dock,) = closeables
    (widget,) = dock.widgets
    assert isinstance(widget, QtChannelList)


def test_render_channel_controls_qt_placeholder_without_config(qtbot):
    """A dock with nothing to drive still renders, so a later add can fill it."""
    from qtpy.QtWidgets import QLabel

    from cellier.convenience.layout._controls_dock import CHANNEL_PLACEHOLDER

    viewer = Viewer(_AXES, dim="2d")
    viewer.add_multichannel_image(_make_store(2), channel_axis=1, channels=_channels(2))

    container = render_dock(ChannelControls(), viewer, QtLayoutHost(), [])

    assert CHANNEL_PLACEHOLDER in [
        label.text() for label in container.findChildren(QLabel)
    ]
