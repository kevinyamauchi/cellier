"""Test fixtures for Cellier."""

import weakref

import numpy as np
import pytest
import tensorstore as ts


@pytest.fixture(autouse=True)
def _close_canvas_views(monkeypatch):
    """Close every ``CanvasView`` a test creates, once that test finishes.

    A render canvas is a parentless widget owned by the GUI backend, so a test
    that simply drops its controller leaks the canvas, its ``WgpuRenderer``,
    and the whole graph they reach -- only ``close()`` reclaims them.  Left
    alone the leak is cumulative: every canvas any earlier test built stays
    live and keeps drawing (they are created ``update_mode="continuous"``), so
    a full run ends holding ~100 of them.  That both starves the later tests
    (``tests/render`` slows to a crawl on CI) and leaves torn-down widgets for
    the Qt event loop to trip over (the Windows access violation).

    Tests build controllers ad hoc rather than through a shared fixture, so
    instances are tracked at construction instead of via a fixture handle.
    """
    from cellier.render.canvas_view import CanvasView

    created: list[weakref.ref] = []
    original_init = CanvasView.__init__

    def _tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(weakref.ref(self))

    monkeypatch.setattr(CanvasView, "__init__", _tracking_init)

    yield

    for ref in created:
        view = ref()
        if view is not None:
            view.close()


@pytest.fixture
def small_zarr_store(tmp_path):
    """A minimal 2-level multiscale zarr v3 store on disk (zeros, float32).

    Shared across the ``gui`` and ``convenience`` suites. ``tests/render`` keeps
    its own local copy (with extra render-specific data fixtures alongside it).
    """
    for name, shape in [("s0", (8, 8, 8)), ("s1", (4, 4, 4))]:
        level_path = tmp_path / name
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(level_path)},
            "metadata": {
                "shape": list(shape),
                "data_type": "float32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [4, 4, 4]},
                },
            },
            "create": True,
            "delete_existing": True,
        }
        store = ts.open(spec).result()
        store[...].write(np.zeros(shape, dtype=np.float32)).result()

    return tmp_path
