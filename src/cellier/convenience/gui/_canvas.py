"""Canvas widget builder for the cellier Viewer convenience layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from cellier.controller import CellierController
    from cellier.convenience._hosts import LayoutHost
    from cellier.convenience._viewer import Viewer
    from cellier.gui.anywidget import ControlPanel
    from cellier.gui.qt import QtCanvasWidget
    from cellier.scene.scene import Scene


def _resolve_gui(viewer: Viewer, gui: str | None) -> str:
    """Resolve the requested *gui*, defaulting to and validating against viewer.gui."""
    if gui is None:
        return viewer.gui
    if gui != viewer.gui:
        raise ValueError(
            f"gui={gui!r} conflicts with viewer.gui={viewer.gui!r}. "
            "Build the canvas with the same gui the viewer was created with."
        )
    return gui


@dataclass
class AnywidgetCanvasView:
    """An anywidget canvas leaf plus its control panel leaf.

    Returned by :func:`build_canvas_widget` for ``gui="anywidget"``.  Holds the
    two host-uniform leaves and knows how to arrange them through an injected
    :class:`~cellier.convenience._hosts.LayoutHost` (canvas above controls).

    Attributes
    ----------
    canvas : object
        The ``rendercanvas`` anywidget canvas.
    controls : ControlPanel
        The composite control panel (dims sliders in Phase 1).
    """

    canvas: object
    controls: ControlPanel

    def compose(self, host: LayoutHost) -> object:
        """Arrange the canvas above the controls via *host*.

        The column is centred on the cross-axis so the fixed-width canvas sits
        centred over the (wider) control panel.
        """
        return host.stack(
            [host.leaf(self.canvas), host.leaf(self.controls)], align="center"
        )

    def close(self) -> None:
        """Unsubscribe the control panel from the bus."""
        self.controls.close()


def canvas_widget_for_scene(
    controller: CellierController,
    scene: Scene,
    axis_ranges: dict[int, tuple[float, float]],
    *,
    render_modes: set[str] | None = None,
    initial_dim: str | None = None,
    fov: float = 70.0,
    depth_range_3d: tuple[float, float] = (1.0, 8000.0),
    depth_range_2d: tuple[float, float] = (-500.0, 500.0),
) -> QtCanvasWidget:
    """Build a wired ``QtCanvasWidget`` for *scene*, reusing any existing canvas.

    If *scene* already has a canvas (e.g. one restored by ``from_file``), that
    canvas -- with its restored camera state -- is reused.  Otherwise a new
    canvas is created.  The returned widget's dims sliders are connected to the
    controller event bus.

    Parameters
    ----------
    controller : CellierController
        The controller owning *scene*.
    scene : Scene
        The scene whose canvas this widget controls.
    axis_ranges : dict[int, tuple[float, float]]
        Mapping of axis index to ``(world_min, world_max)`` for slider ranges.
    render_modes : set[str] or None
        Camera modes to prepare when creating a new canvas.  Defaults to the
        scene's own ``render_modes``.  Ignored when reusing an existing canvas.
    initial_dim : str or None
        Active mode for a newly created canvas.  Inferred from the scene's
        displayed axes when ``None``.
    fov : float
        Vertical field of view in degrees for a new 3D camera.  Default ``70``.
    depth_range_3d : tuple[float, float]
        ``(near, far)`` clip distances for a new 3D camera.
    depth_range_2d : tuple[float, float]
        ``(near, far)`` clip distances for a new 2D camera.

    Returns
    -------
    QtCanvasWidget
    """
    from cellier.gui.qt import QtCanvasWidget

    canvas_ids = controller.get_canvas_ids(scene.id)
    if not canvas_ids:
        controller.add_canvas(
            scene.id,
            render_modes=render_modes
            if render_modes is not None
            else set(scene.render_modes),
            initial_dim=initial_dim,
            fov=fov,
            depth_range_3d=depth_range_3d,
            depth_range_2d=depth_range_2d,
        )
        canvas_ids = controller.get_canvas_ids(scene.id)

    canvas_view = controller.get_canvas_view(canvas_ids[-1])
    canvas_widget = QtCanvasWidget.from_scene_and_canvas(
        scene,
        canvas_view,
        axis_ranges,
    )
    controller.connect_widget(
        canvas_widget.dims_sliders,
        subscription_specs=canvas_widget.dims_sliders.subscription_specs(),
    )
    return canvas_widget


def build_canvas_widget(
    viewer: Viewer,
    axis_ranges: dict[int, tuple[float, float]],
    *,
    gui: Literal["qt", "anywidget"] | None = None,
    render_modes: set[str] | None = None,
    initial_dim: str | None = None,
    fov: float = 70.0,
    depth_range_3d: tuple[float, float] = (1.0, 8000.0),
    depth_range_2d: tuple[float, float] = (-500.0, 500.0),
    canvas_size: tuple[int, int] | None = None,
) -> QtCanvasWidget | AnywidgetCanvasView:
    """Build a canvas widget with wired dims sliders for the viewer.

    Creates a canvas attached to the viewer's scene and connects its dims
    sliders to the controller event bus.  For ``gui="qt"`` this returns a
    ``QtCanvasWidget`` (canvas above a dims slider panel); for
    ``gui="anywidget"`` it returns an :class:`AnywidgetCanvasView` holding the
    canvas and control-panel leaves plus a ``compose(host)`` method.

    Parameters
    ----------
    viewer : Viewer
        The viewer to attach the canvas to.
    axis_ranges : dict[int, tuple[float, float]]
        Mapping of axis index to ``(world_min, world_max)`` used to set the
        slider ranges.  Typically obtained from
        :func:`cellier.convenience.axis_ranges_from_viewer`.
    gui : "qt", "anywidget", or None
        GUI toolkit.  Defaults to ``viewer.gui`` when ``None``; raises if it
        conflicts with ``viewer.gui``.
    render_modes : set[str] or None
        Which camera modes to prepare on the canvas.  Defaults to the scene's
        own ``render_modes``.
    initial_dim : str or None
        Which mode is active first.  Inferred from the scene's current
        ``displayed_axes`` when ``None``.
    fov : float
        Vertical field-of-view in degrees for the 3D perspective camera.
        Default ``70.0``.
    depth_range_3d : tuple[float, float]
        ``(near, far)`` clip distances for the 3D camera.
        Default ``(1.0, 8000.0)``.
    depth_range_2d : tuple[float, float]
        ``(near, far)`` clip distances for the 2D camera.
        Default ``(-500.0, 500.0)``.
    canvas_size : tuple[int, int] or None
        Initial CSS pixel size for the anywidget canvas.  Ignored for the Qt
        gui.  Defaults to ``(600, 600)`` for the anywidget gui.

    Returns
    -------
    QtCanvasWidget or AnywidgetCanvasView
        For ``gui="qt"``, a widget composing the render surface and dims
        slider panel (embed ``canvas_widget.widget``).  For
        ``gui="anywidget"``, an :class:`AnywidgetCanvasView`.

    Raises
    ------
    ValueError
        If *gui* conflicts with ``viewer.gui`` or is not recognised.
    """
    gui = _resolve_gui(viewer, gui)
    if gui == "qt":
        return _build_qt_canvas_widget(
            viewer,
            axis_ranges,
            render_modes=render_modes,
            initial_dim=initial_dim,
            fov=fov,
            depth_range_3d=depth_range_3d,
            depth_range_2d=depth_range_2d,
        )
    elif gui == "anywidget":
        return anywidget_canvas_view_for_scene(
            viewer.controller,
            viewer.scene,
            axis_ranges,
            render_modes=render_modes,
            initial_dim=initial_dim,
            fov=fov,
            depth_range_3d=depth_range_3d,
            depth_range_2d=depth_range_2d,
            canvas_size=canvas_size,
        )
    raise ValueError(f"Unknown gui {gui!r}. Expected 'qt' or 'anywidget'.")


def _build_qt_canvas_widget(
    viewer: Viewer,
    axis_ranges: dict[int, tuple[float, float]],
    *,
    render_modes: set[str] | None = None,
    initial_dim: str | None = None,
    fov: float = 70.0,
    depth_range_3d: tuple[float, float] = (1.0, 8000.0),
    depth_range_2d: tuple[float, float] = (-500.0, 500.0),
) -> QtCanvasWidget:
    """Qt implementation of :func:`build_canvas_widget`."""
    return canvas_widget_for_scene(
        viewer.controller,
        viewer.scene,
        axis_ranges,
        render_modes=render_modes,
        initial_dim=initial_dim,
        fov=fov,
        depth_range_3d=depth_range_3d,
        depth_range_2d=depth_range_2d,
    )


def anywidget_canvas_view_for_scene(
    controller: CellierController,
    scene: Scene,
    axis_ranges: dict[int, tuple[float, float]],
    *,
    render_modes: set[str] | None = None,
    initial_dim: str | None = None,
    fov: float = 70.0,
    depth_range_3d: tuple[float, float] = (1.0, 8000.0),
    depth_range_2d: tuple[float, float] = (-500.0, 500.0),
    canvas_size: tuple[int, int] | None = None,
    non_displayed: tuple[int, ...] = (),
) -> AnywidgetCanvasView:
    """Build a wired :class:`AnywidgetCanvasView` for *scene*, reusing any canvas.

    Mirrors :func:`canvas_widget_for_scene` for the anywidget gui: ensures a
    canvas exists (the controller's ``gui == "anywidget"`` makes it a
    rendercanvas anywidget), builds a ``ControlPanel`` from the scene, wires it
    to the bus, and returns the two leaves.

    Parameters
    ----------
    controller : CellierController
        The controller owning *scene* (must have ``gui="anywidget"``).
    scene : Scene
        The scene whose canvas + dims this view controls.
    axis_ranges : dict[int, tuple[float, float]]
        Mapping of axis index to ``(world_min, world_max)`` for slider ranges.
    render_modes, initial_dim, fov, depth_range_3d, depth_range_2d, canvas_size
        Forwarded to :meth:`CellierController.add_canvas` when a new canvas is
        created.  Ignored when reusing an existing canvas.
    non_displayed : tuple[int, ...]
        Axes to exclude from the sliders regardless of dims state.

    Returns
    -------
    AnywidgetCanvasView
    """
    from cellier.gui.anywidget import ControlPanel

    canvas_ids = controller.get_canvas_ids(scene.id)
    if not canvas_ids:
        controller.add_canvas(
            scene.id,
            render_modes=render_modes
            if render_modes is not None
            else set(scene.render_modes),
            initial_dim=initial_dim,
            fov=fov,
            depth_range_3d=depth_range_3d,
            depth_range_2d=depth_range_2d,
            canvas_size=canvas_size,
        )
        canvas_ids = controller.get_canvas_ids(scene.id)

    canvas_view = controller.get_canvas_view(canvas_ids[-1])
    panel = ControlPanel.from_scene(scene, axis_ranges, non_displayed=non_displayed)
    controller.connect_widget(panel, subscription_specs=panel.subscription_specs())
    return AnywidgetCanvasView(canvas=canvas_view.widget, controls=panel)
