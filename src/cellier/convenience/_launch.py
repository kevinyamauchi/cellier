"""Blocking and non-blocking launchers for a cellier Viewer.

Qt imports are lazy so that importing this module does not require a Qt
installation.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cellier.convenience._ortho_viewer import OrthoViewer
    from cellier.convenience._viewer import Viewer

    ViewerLike = Viewer | OrthoViewer

FitMode = Literal["ready", "immediate", "none"]


def launch(
    viewer: ViewerLike,
    window: object,
    *,
    fit: FitMode = "ready",
    on_ready: Callable[[], None] | None = None,
    handle_sigint: bool = True,
) -> None:
    """Show *window* and block until it is closed.

    Creates a ``QApplication`` if one does not already exist, then starts the
    Qt + asyncio event loop via ``PySide6.QtAsyncio``.  Intended for scripts
    and the CLI.  For interactive / Jupyter use, call :func:`show` instead.

    The initial camera fit and data load are deferred to the canvas's first
    rendered frame (see :meth:`CellierController.on_canvas_first_frame`), so
    they run only once the canvas has its final size and the event loop is
    live — no timer required.

    Parameters
    ----------
    viewer : Viewer
        The viewer whose controller is used for ``fit_camera`` /
        ``reslice_scene``.
    window : QMainWindow or QWidget
        The top-level Qt window to show.
    fit : "ready", "immediate", or "none"
        Camera-fit policy applied at startup.  See :func:`_init_view`.
    on_ready : Callable[[], None] or None
        Optional zero-argument callback fired once, after every scene's startup
        data has committed to the GPU.  Fires in addition to any callbacks
        registered via ``viewer.on_ready``.
    handle_sigint : bool
        Pass ``True`` (default) to let QtAsyncio install a ``SIGINT`` handler
        so that Ctrl-C closes the window cleanly.
    """
    import PySide6.QtAsyncio as QtAsyncio
    from PySide6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([sys.argv[0]])

    window.show()  # type: ignore[attr-defined]
    # _init_view only subscribes to the first-frame event and requests a draw;
    # the actual fit + reslice run inside that callback, which fires once
    # QtAsyncio.run() below starts the loop and the canvas paints.  No timer is
    # needed because the work is gated on a real render event, not a delay.
    _init_view(viewer, fit=fit, on_ready=on_ready)
    QtAsyncio.run(handle_sigint=handle_sigint)


def show(
    viewer: ViewerLike,
    window: object,
    *,
    fit: FitMode = "ready",
    on_ready: Callable[[], None] | None = None,
) -> None:
    """Show *window* without blocking.

    Requires a Qt event loop that is already running (e.g. inside IPython /
    Jupyter after ``%gui qt``).  The initial camera fit and data load are
    deferred to the canvas's first rendered frame.

    Parameters
    ----------
    viewer : Viewer
        The viewer whose controller is used for ``fit_camera`` /
        ``reslice_scene``.
    window : QMainWindow or QWidget
        The top-level Qt window to show.
    fit : "ready", "immediate", or "none"
        Camera-fit policy applied at startup.  See :func:`_init_view`.
    on_ready : Callable[[], None] or None
        Optional zero-argument callback fired once, after every scene's startup
        data has committed to the GPU.

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
    _init_view(viewer, fit=fit, on_ready=on_ready)


class DisplayHandle:
    """Inert teardown handle returned by :func:`display`.

    Rendering is performed imperatively by the host inside ``present()``; this
    handle is the cell's return value but has no representation of its own (its
    ``_repr_mimebundle_`` is empty), so the viewer is rendered exactly once.

    Call :meth:`close` to unsubscribe the control panel(s) from the bus and
    cancel any pending slices -- e.g. before re-running a cell so the prior
    panel and in-flight reads do not leak.
    """

    def __init__(self, viewer: ViewerLike, view: object) -> None:
        self._viewer = viewer
        self._view = view
        self._closed = False

    def close(self) -> None:
        """Tear down the controls and cancel pending slices (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._view.close()
        scenes = getattr(self._viewer, "scenes", None)
        scene_list = scenes.values() if scenes is not None else [self._viewer.scene]
        for scene in scene_list:
            self._viewer.controller.cancel_pending_slices(scene.id)

    def _repr_mimebundle_(self, **kwargs):
        # Inert: the host already rendered the viewer imperatively in present(),
        # so the handle itself must not produce a second copy in the cell.
        return {}

    def __repr__(self) -> str:
        # IPython's plain-text formatter falls back to repr() even when the mime
        # bundle is empty; blank it so no stray ``Out[]`` text appears under the
        # viewer.
        return ""


def display(
    viewer: ViewerLike,
    view: object,
    *,
    controls: Iterable[object] = (),
    controls_position: Literal["top", "bottom"] = "top",
    fit: FitMode = "ready",
    on_ready: Callable[[], None] | None = None,
    host: str | None = None,
) -> object:
    """Compose and present an anywidget viewer non-blockingly.

    The notebook counterpart of :func:`launch`.  Resolves the anywidget host
    (Jupyter or marimo), composes the *controls* above or below the *view*
    through the host's :class:`~cellier.convenience._hosts.LayoutHost`, renders
    the composed root, and arms first-frame startup.  No ``QApplication`` /
    ``QtAsyncio`` is involved -- slicing runs on the host's own asyncio loop.

    Rendering is done imperatively inside ``host.present()`` (exactly once,
    before ``_init_view`` arms the first-frame fit so the canvas is mounted and
    paints), and this function returns an inert :class:`DisplayHandle` rather
    than the widget, so the cell shows a single copy regardless of where
    ``display()`` sits in it.

    Parameters
    ----------
    viewer : Viewer or OrthoViewer
        The viewer whose controller drives fit / reslice.
    view : AnywidgetCanvasView or OrthoAnywidgetCanvases
        The object returned by ``build_canvas_widget`` /
        ``build_ortho_grid_widget`` for ``gui="anywidget"``.  Must expose
        ``compose(host)`` and ``close()``.
    controls : iterable of anywidget
        Extra app-level controls (e.g. ``make_dim_toggle(viewer)``) placed
        above or below the view (see *controls_position*).
    controls_position : "top" or "bottom"
        Whether the *controls* are stacked above (``"top"``, default) or below
        (``"bottom"``) the view.
    fit : "ready", "immediate", or "none"
        Camera-fit policy applied at startup.  See :func:`_init_view`.
    on_ready : Callable[[], None] or None
        Optional zero-argument callback fired once every scene's startup data
        has committed to the GPU.
    host : "jupyter", "marimo", or None
        Explicit host override; auto-detected when ``None``.

    Returns
    -------
    object
        For imperative hosts (Jupyter) an inert :class:`DisplayHandle` whose
        :meth:`DisplayHandle.close` tears down the controls and cancels pending
        slices.  For return-value hosts (marimo) the host-native renderable
        (so the cell renders it), with a best-effort ``close`` attached.
    """
    from cellier.convenience._hosts import resolve_host

    if controls_position not in ("top", "bottom"):
        raise ValueError(
            f"controls_position must be 'top' or 'bottom', got {controls_position!r}."
        )

    resolved_host = resolve_host(host)
    control_leaves = [resolved_host.leaf(control) for control in controls]
    view_leaf = view.compose(resolved_host)
    leaves = (
        [view_leaf, *control_leaves]
        if controls_position == "bottom"
        else [*control_leaves, view_leaf]
    )

    # Imperative hosts (Jupyter) render in present() and return None; return-value
    # hosts (marimo) return the renderable for display() to yield as the cell
    # output (see LayoutHost.present).  Centre the column so controls line up
    # under the (fixed-width) viewer.
    cell_value = resolved_host.present(resolved_host.stack(leaves, align="center"))

    # _init_view only subscribes to the first-frame event and requests a draw;
    # the actual fit + reslice run inside that callback once the host loop ticks
    # and the canvas paints its first frame (after the cell value is rendered).
    _init_view(viewer, fit=fit, on_ready=on_ready)

    handle = DisplayHandle(viewer, view)
    if cell_value is None:
        # Imperative host already rendered; the inert handle is the cell value.
        return handle
    # Return-value host: yield the renderable, attaching teardown best-effort.
    try:
        cell_value.close = handle.close  # type: ignore[attr-defined]
    except Exception:
        pass
    return cell_value


def _init_view(
    viewer: ViewerLike,
    *,
    fit: FitMode = "ready",
    on_ready: Callable[[], None] | None = None,
) -> None:
    """Arm first-frame startup (fit + reslice) for every scene.

    Supports both the single-scene :class:`Viewer` (which exposes ``scene``)
    and the multi-panel :class:`OrthoViewer` (which exposes ``scenes``).

    The fit policy controls how the camera is framed:

    * ``"ready"`` (default) — fit on the first frame, then **re-fit** once the
      scene's data has committed to the GPU.  The re-fit is what makes geometry
      visuals (mesh, points, lines) frame correctly, since their bounding box
      is only known after their data loads.  For image visuals the re-fit is a
      no-op (their bounds are pinned at construction).
    * ``"immediate"`` — fit on the first frame only; no re-fit after load.
    * ``"none"`` — never fit; just trigger the initial data load.

    Any callbacks registered on the viewer via ``viewer.on_ready`` plus the
    *on_ready* argument fire once, after the *last* scene becomes ready.

    Camera-driven reslicing is suppressed for the duration of the startup load
    and restored once every scene is ready.  Without this, the camera move
    produced by the initial ``fit_camera`` would schedule a settle reslice that
    cancels the in-flight startup reads, which could starve the readiness
    callback for slow (e.g. remote) data.
    """
    controller = viewer.controller
    scenes = getattr(viewer, "scenes", None)
    scene_list = list(scenes.values() if scenes is not None else [viewer.scene])

    user_callbacks = list(getattr(viewer, "_ready_callbacks", []))
    if on_ready is not None:
        user_callbacks.append(on_ready)

    # Suppress camera-settle reslices during startup so the initial load is a
    # single, un-cancelled generation; restore the prior setting once ready.
    prev_reslice_enabled = controller.camera_reslice_enabled
    controller.camera_reslice_enabled = False

    aggregate = {"remaining": len(scene_list), "fired": False}

    def _finish_startup() -> None:
        aggregate["fired"] = True
        controller.camera_reslice_enabled = prev_reslice_enabled
        for cb in user_callbacks:
            cb()

    def _scene_ready() -> None:
        aggregate["remaining"] -= 1
        if aggregate["remaining"] <= 0 and not aggregate["fired"]:
            _finish_startup()

    if not scene_list:
        _finish_startup()
        return

    for scene in scene_list:
        canvas_ids = controller.get_canvas_ids(scene.id)
        if not canvas_ids:
            # No canvas attached yet; still load data so the model is populated.
            controller.reslice_scene(scene.id, on_ready=_scene_ready)
            continue

        def _start(s=scene) -> None:
            if fit != "none":
                controller.fit_camera(s.id)

            def _ready(s=s) -> None:
                if fit == "ready":
                    controller.fit_camera(s.id)
                _scene_ready()

            controller.reslice_scene(s.id, on_ready=_ready)

        controller.on_canvas_first_frame(canvas_ids[0], _start, owner_id=controller._id)
