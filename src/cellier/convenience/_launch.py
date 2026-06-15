"""Blocking and non-blocking launchers for a cellier Viewer.

Qt imports are lazy so that importing this module does not require a Qt
installation.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cellier.convenience._ortho_viewer import OrthoViewer
    from cellier.convenience._viewer import Viewer

    ViewerLike = Viewer | OrthoViewer


def launch(
    viewer: ViewerLike,
    window: object,
    *,
    handle_sigint: bool = True,
) -> None:
    """Show *window* and block until it is closed.

    Creates a ``QApplication`` if one does not already exist, then starts the
    Qt + asyncio event loop via ``PySide6.QtAsyncio``.  Intended for scripts
    and the CLI.  For interactive / Jupyter use, call :func:`show` instead.

    After showing the window this function calls
    ``controller.fit_camera`` and ``controller.reslice_scene`` so the initial
    view is centred and populated.

    Parameters
    ----------
    viewer : Viewer
        The viewer whose controller is used for ``fit_camera`` /
        ``reslice_scene``.
    window : QMainWindow or QWidget
        The top-level Qt window to show.
    handle_sigint : bool
        Pass ``True`` (default) to let QtAsyncio install a ``SIGINT`` handler
        so that Ctrl-C closes the window cleanly.
    """
    import PySide6.QtAsyncio as QtAsyncio
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([sys.argv[0]])

    window.show()  # type: ignore[attr-defined]
    # reslice_scene submits asynchronous reads that only run once the event
    # loop is live.  QtAsyncio.run() below starts that loop, so the initial
    # fit + reslice must be deferred to the first loop iteration; calling them
    # now would queue a fetch that never completes until the next reslice
    # (e.g. a slider move).  The zero-delay timer also lets the canvas reach
    # its final size before fit_camera reads the scene bounds.
    QTimer.singleShot(0, lambda: _init_view(viewer))
    QtAsyncio.run(handle_sigint=handle_sigint)


def show(viewer: ViewerLike, window: object) -> None:
    """Show *window* without blocking.

    Requires a Qt event loop that is already running (e.g. inside IPython /
    Jupyter after ``%gui qt``).  After showing the window this function calls
    ``controller.fit_camera`` and ``controller.reslice_scene``.

    Parameters
    ----------
    viewer : Viewer
        The viewer whose controller is used for ``fit_camera`` /
        ``reslice_scene``.
    window : QMainWindow or QWidget
        The top-level Qt window to show.

    Raises
    ------
    RuntimeError
        If no ``QApplication`` instance is running.
    """
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:
        raise RuntimeError(
            "No Qt event loop is running. "
            "Use launch() for scripts, or run inside IPython/Jupyter with %gui qt."
        )
    window.show()  # type: ignore[attr-defined]
    # A loop is already running here, so the initial reslice can run inline.
    _init_view(viewer)


def _init_view(viewer: ViewerLike) -> None:
    """Fit the camera and trigger the initial reslice for every scene.

    Supports both the single-scene :class:`Viewer` (which exposes ``scene``)
    and the multi-panel :class:`OrthoViewer` (which exposes ``scenes``).
    """
    controller = viewer.controller
    scenes = getattr(viewer, "scenes", None)
    scene_objects = scenes.values() if scenes is not None else [viewer.scene]
    for scene in scene_objects:
        controller.fit_camera(scene.id)
        controller.reslice_scene(scene.id)
