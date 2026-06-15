"""Canvas widget builder for the cellier Viewer convenience layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from cellier.controller import CellierController
    from cellier.convenience._viewer import Viewer
    from cellier.gui import QtCanvasWidget
    from cellier.scene.scene import Scene


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
    from cellier.gui import QtCanvasWidget

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
    backend: Literal["qt"] = "qt",
    render_modes: set[str] | None = None,
    initial_dim: str | None = None,
    fov: float = 70.0,
    depth_range_3d: tuple[float, float] = (1.0, 8000.0),
    depth_range_2d: tuple[float, float] = (-500.0, 500.0),
) -> QtCanvasWidget:
    """Build a canvas widget with wired dims sliders for the viewer.

    Creates a canvas attached to the viewer's scene, wraps it in a
    ``QtCanvasWidget`` (canvas + dims slider panel), and connects the dims
    sliders to the controller event bus.

    Parameters
    ----------
    viewer : Viewer
        The viewer to attach the canvas to.
    axis_ranges : dict[int, tuple[float, float]]
        Mapping of axis index to ``(world_min, world_max)`` used to set the
        slider ranges.  Typically obtained from
        :func:`cellier.convenience.axis_ranges_from_viewer`.
    backend : "qt"
        Rendering backend.  Only ``"qt"`` is currently supported.
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

    Returns
    -------
    QtCanvasWidget
        A widget that composes the render surface and the dims slider panel.
        Embed ``canvas_widget.widget`` in your Qt layout.

    Raises
    ------
    ValueError
        If *backend* is not ``"qt"``.
    """
    if backend != "qt":
        raise ValueError(f"Unknown backend {backend!r}. Only 'qt' is supported.")
    return _build_qt_canvas_widget(
        viewer,
        axis_ranges,
        render_modes=render_modes,
        initial_dim=initial_dim,
        fov=fov,
        depth_range_3d=depth_range_3d,
        depth_range_2d=depth_range_2d,
    )


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
