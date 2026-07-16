"""Tests for ``cellier.convenience._sidecar.resolve_sidecar``.

The validation branches (ref type errors) run before any ``sidecar`` import, so
they are dependency-free. The success paths and the ``ImportError`` message
path do touch the optional ``sidecar`` package.
"""

from __future__ import annotations

import sys

import pytest

from cellier.convenience._launch import DisplayHandle
from cellier.convenience._sidecar import SidecarOptions, resolve_sidecar

# ---------------------------------------------------------------------------
# Validation branches (no sidecar import reached)
# ---------------------------------------------------------------------------


def test_ref_as_sidecar_options_raises_type_error():
    opts = SidecarOptions(ref=SidecarOptions())
    with pytest.raises(TypeError, match="must be a DisplayHandle"):
        resolve_sidecar(opts)


def test_ref_handle_without_sidecar_raises_value_error():
    handle = DisplayHandle(object(), object())  # _sidecar is None
    with pytest.raises(ValueError, match="not\\s+itself displayed with sidecar"):
        resolve_sidecar(SidecarOptions(ref=handle))


# ---------------------------------------------------------------------------
# ImportError path
# ---------------------------------------------------------------------------


def test_missing_sidecar_package_raises_import_error(monkeypatch):
    # Force ``import sidecar`` inside resolve_sidecar to fail.
    monkeypatch.setitem(sys.modules, "sidecar", None)
    with pytest.raises(ImportError, match="requires the optional 'sidecar'"):
        resolve_sidecar(True)


# ---------------------------------------------------------------------------
# Success paths (sidecar installed)
# ---------------------------------------------------------------------------

pytest.importorskip("sidecar")


def test_true_uses_defaults():
    s = resolve_sidecar(True)
    assert s.title == "Cellier"
    assert s.anchor == "right"


def test_dict_is_coerced():
    s = resolve_sidecar({"title": "X", "anchor": "split-right"})
    assert s.title == "X"
    assert s.anchor == "split-right"


def test_options_instance_passes_through():
    s = resolve_sidecar(SidecarOptions(title="Y", anchor="tab-after"))
    assert s.title == "Y"
    assert s.anchor == "tab-after"


def test_title_none_falls_back_to_sidecar_default():
    # title=None must be omitted (Sidecar's title trait is non-nullable), so the
    # package's own default title is used rather than raising.
    s = resolve_sidecar(SidecarOptions(title=None))
    assert s.title == "Sidecar"
