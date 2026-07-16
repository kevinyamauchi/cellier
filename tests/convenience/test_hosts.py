"""Tests for ``cellier.convenience._hosts`` (MarimoHost + host detection).

The ``MarimoHost`` methods are driven with a fake ``mo`` module so no live
marimo runtime is needed; the one real bit of logic is the px->rem ``gap``
conversion. ``resolve_host`` and the two detector helpers are covered by
monkeypatching the detectors / import machinery.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from cellier.convenience import _hosts
from cellier.convenience._hosts import (
    JupyterHost,
    MarimoHost,
    _ipython_running,
    _marimo_running,
    resolve_host,
)


def _fake_mo():
    """A stand-in ``marimo`` module recording vstack/hstack/anywidget calls."""

    def vstack(items, align=None, **kw):
        return ("vstack", list(items), align, kw)

    def hstack(items, align=None, **kw):
        return ("hstack", list(items), align, kw)

    def anywidget(widget):
        return ("leaf", widget)

    return SimpleNamespace(
        vstack=vstack, hstack=hstack, ui=SimpleNamespace(anywidget=anywidget)
    )


def _marimo_host_with_fake_mo() -> MarimoHost:
    host = MarimoHost.__new__(MarimoHost)  # bypass __init__ (no real marimo import)
    host._mo = _fake_mo()
    return host


# ---------------------------------------------------------------------------
# MarimoHost delegation + the px->rem gap conversion (the real logic)
# ---------------------------------------------------------------------------


def test_marimo_stack_converts_gap_px_to_rem():
    host = _marimo_host_with_fake_mo()
    result = host.stack(["a", "b"], direction="h", gap=32)
    assert result[0] == "hstack"
    assert result[3] == {"gap": 2.0}  # 32 px / 16 == 2 rem


def test_marimo_stack_omits_gap_when_none():
    host = _marimo_host_with_fake_mo()
    result = host.stack(["a"], direction="v")
    assert result[0] == "vstack"
    assert result[3] == {}  # no gap kwarg


def test_marimo_leaf_wraps_with_ui_anywidget():
    host = _marimo_host_with_fake_mo()
    assert host.leaf("w") == ("leaf", "w")


def test_marimo_grid_nests_hstacks_in_a_vstack():
    host = _marimo_host_with_fake_mo()
    result = host.grid([["a", "b"], ["c"]])
    assert result[0] == "vstack"
    rows = result[1]
    assert [r[0] for r in rows] == ["hstack", "hstack"]


def test_marimo_present_returns_root_unchanged():
    host = _marimo_host_with_fake_mo()
    root = object()
    assert host.present(root) is root


# ---------------------------------------------------------------------------
# Detector helpers
# ---------------------------------------------------------------------------


def test_marimo_running_true(monkeypatch):
    marimo = pytest.importorskip("marimo")
    monkeypatch.setattr(marimo, "running_in_notebook", lambda: True)
    assert _marimo_running() is True


def test_marimo_running_false_when_import_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, "marimo", None)
    assert _marimo_running() is False


def test_ipython_running_true(monkeypatch):
    ipython = pytest.importorskip("IPython")
    monkeypatch.setattr(ipython, "get_ipython", lambda: object())
    assert _ipython_running() is True


def test_ipython_running_false(monkeypatch):
    ipython = pytest.importorskip("IPython")
    monkeypatch.setattr(ipython, "get_ipython", lambda: None)
    assert _ipython_running() is False


# ---------------------------------------------------------------------------
# resolve_host
# ---------------------------------------------------------------------------


def test_resolve_host_explicit_jupyter():
    assert isinstance(resolve_host("jupyter"), JupyterHost)


def test_resolve_host_explicit_marimo():
    pytest.importorskip("marimo")
    assert isinstance(resolve_host("marimo"), MarimoHost)


def test_resolve_host_autodetect_marimo(monkeypatch):
    pytest.importorskip("marimo")
    monkeypatch.setattr(_hosts, "_marimo_running", lambda: True)
    assert isinstance(resolve_host(None), MarimoHost)


def test_resolve_host_autodetect_jupyter(monkeypatch):
    monkeypatch.setattr(_hosts, "_marimo_running", lambda: False)
    monkeypatch.setattr(_hosts, "_ipython_running", lambda: True)
    assert isinstance(resolve_host(None), JupyterHost)


def test_resolve_host_none_detected_raises(monkeypatch):
    monkeypatch.setattr(_hosts, "_marimo_running", lambda: False)
    monkeypatch.setattr(_hosts, "_ipython_running", lambda: False)
    with pytest.raises(RuntimeError, match="No anywidget host detected"):
        resolve_host(None)
