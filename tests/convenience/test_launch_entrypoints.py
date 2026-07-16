"""Guard-clause tests for the Qt/notebook entry points in ``_launch``.

The full ``launch``/``show``/``display`` bodies enter or require a live event
loop, so these cover only the cheap, high-value guards and ``_resolve_qt_window``
(which is pure dispatch).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cellier.convenience._launch import _resolve_qt_window
from cellier.convenience.layout._spec import Layout


def test_show_without_qapplication_raises(monkeypatch):
    pytest.importorskip("PySide6")
    from cellier.convenience._launch import show

    # Force the "no running Qt app" branch regardless of session state.
    monkeypatch.setattr("PySide6.QtWidgets.QApplication.instance", lambda: None)
    with pytest.raises(RuntimeError, match="No Qt event loop"):
        show(SimpleNamespace(), object())


def test_display_sidecar_requires_jupyter_host():
    pytest.importorskip("marimo")
    from cellier.convenience._launch import display

    # host="marimo" + sidecar=... must raise before any rendering happens.
    with pytest.raises(RuntimeError, match="requires the Jupyter host"):
        display(SimpleNamespace(), object(), host="marimo", sidecar=True)


def test_resolve_qt_window_passthrough():
    window = object()
    assert _resolve_qt_window(window, SimpleNamespace()) is window


def test_resolve_qt_window_renders_layout(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "cellier.convenience.layout._qt_renderer.render_qt",
        lambda layout, viewer: sentinel,
    )
    layout = Layout(center=object())
    assert _resolve_qt_window(layout, SimpleNamespace()) is sentinel
