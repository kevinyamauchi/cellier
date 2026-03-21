"""CanvasView — owns one rendered canvas with camera and controller."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import pygfx as gfx
from rendercanvas.qt import QRenderWidget

from cellier.v2.render._requests import DimsState, ReslicingRequest

if TYPE_CHECKING:
    from collections.abc import Callable

    from PySide6.QtWidgets import QWidget


class CanvasView:
    """Owns one rendered canvas: widget, renderer, camera, and controller.

    Responsible for rendering one scene from one camera viewpoint.
    ``CanvasView`` does not hold a direct reference to the scene graph;
    instead it receives a ``get_scene_fn`` callable that is invoked each
    frame so ownership of the scene stays with ``SceneManager``.

    Camera sync loop prevention is implemented via the
    ``_applying_model_state`` flag, which suppresses re-entrant
    ``_on_controller_event`` calls during programmatic camera updates.
    This guard is dormant until EventBus wiring is added.

    Parameters
    ----------
    canvas_id : UUID
        Unique identifier for this canvas.
    scene_id : UUID
        ID of the scene this canvas renders.
    get_scene_fn : Callable[[UUID], gfx.Scene]
        Called each frame to retrieve the current scene.  Provided by
        ``RenderManager`` at construction time.
    parent : QWidget or None
        Parent widget for the underlying ``QRenderWidget``.
    fov : float
        Vertical field of view in degrees.
    depth_range : tuple[float, float]
        Near and far clip distances ``(near, far)``.
    """

    def __init__(
        self,
        canvas_id: UUID,
        scene_id: UUID,
        get_scene_fn: Callable[[UUID], gfx.Scene],
        parent: QWidget | None = None,
        fov: float = 70.0,
        depth_range: tuple[float, float] = (1.0, 8000.0),
    ) -> None:
        self._canvas_id = canvas_id
        self._scene_id = scene_id
        self._get_scene_fn = get_scene_fn
        self._applying_model_state: bool = False

        self._canvas = QRenderWidget(parent=parent, update_mode="continuous")
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._camera = gfx.PerspectiveCamera(fov, 16 / 9, depth_range=depth_range)
        self._controller = gfx.OrbitController(
            camera=self._camera, register_events=self._renderer
        )
        # future: self._renderer.add_event_handler(
        #     self._on_controller_event, "pointer_move", "wheel"
        # )
        self._canvas.request_draw(self._draw_frame)

    @property
    def canvas_id(self) -> UUID:
        """Unique identifier for this canvas."""
        return self._canvas_id

    @property
    def scene_id(self) -> UUID:
        """ID of the scene this canvas renders."""
        return self._scene_id

    @property
    def widget(self) -> QRenderWidget:
        """The Qt widget to embed in the application layout."""
        return self._canvas

    def capture_reslicing_request(
        self,
        dims_state: DimsState,
        target_visual_ids: frozenset[UUID] | None = None,
    ) -> ReslicingRequest:
        """Snapshot the current camera state into a ReslicingRequest.

        All array fields are copied.  ``screen_height_px`` is read from
        the canvas at call time and baked into the returned request.

        Parameters
        ----------
        dims_state : DimsState
            Current dimension display state.
        target_visual_ids : frozenset[UUID] or None
            ``None`` reslices all visuals in the scene.

        Returns
        -------
        ReslicingRequest
            Fully populated snapshot with independent array copies.
        """
        _, screen_h = self._canvas.get_logical_size()
        return ReslicingRequest(
            camera_pos=np.array(self._camera.world.position, dtype=np.float64),
            frustum_corners=np.asarray(self._camera.frustum, dtype=np.float64).copy(),
            fov_y_rad=float(np.radians(self._camera.fov)),
            screen_height_px=float(screen_h),
            dims_state=dims_state,
            request_id=uuid4(),
            scene_id=self._scene_id,
            target_visual_ids=target_visual_ids,
        )

    def show_object(self, scene: gfx.Scene) -> None:
        """Fit the camera to the scene bounding box.

        Parameters
        ----------
        scene : gfx.Scene
            The scene to fit the camera to.
        """
        self._camera.show_object(scene, view_dir=(-1, -1, -1), up=(0, 0, 1))

    def request_draw(self) -> None:
        """Request a redraw of the canvas."""
        self._canvas.request_draw(self._draw_frame)

    def apply_camera_state(self, request: ReslicingRequest) -> None:
        """Apply a camera snapshot from the model layer (programmatic move).

        The ``_applying_model_state`` guard prevents the resulting camera
        setter calls from firing ``_on_controller_event``, which would
        otherwise cause a feedback loop: model change -> apply to pygfx ->
        controller event -> model change -> ...

        Parameters
        ----------
        request : ReslicingRequest
            Camera snapshot to apply.
        """
        self._applying_model_state = True
        try:
            self._camera.world.position = tuple(request.camera_pos)
        finally:
            self._applying_model_state = False

    def _draw_frame(self) -> None:
        scene = self._get_scene_fn(self._scene_id)
        self._renderer.render(scene, self._camera)

    def _on_controller_event(self, event) -> None:
        if self._applying_model_state:
            return
        # future: self._event_bus.emit(CameraMovedEvent(source="controller", ...))
