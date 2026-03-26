"""CanvasView — owns one rendered canvas with camera and controller."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import pygfx as gfx
from rendercanvas.qt import QRenderWidget

from cellier.v2._state import CameraState
from cellier.v2.events._events import CameraChangedEvent
from cellier.v2.logging import _CAMERA_LOGGER
from cellier.v2.render._requests import DimsState, ReslicingRequest

if TYPE_CHECKING:
    from collections.abc import Callable

    from PySide6.QtWidgets import QWidget

    from cellier.v2.events._bus import EventBus


class CanvasView:
    """Owns one rendered canvas: widget, renderer, camera, and controller.

    Responsible for rendering one scene from one camera viewpoint.
    ``CanvasView`` does not hold a direct reference to the scene graph;
    instead it receives a ``get_scene_fn`` callable that is invoked each
    frame so ownership of the scene stays with ``SceneManager``.

    Camera change detection is implemented by comparing a cached
    ``CameraState`` snapshot each frame in ``_draw_frame``.  The
    ``_applying_model_state`` flag suppresses detection during
    programmatic camera updates to prevent feedback loops.

    Parameters
    ----------
    canvas_id : UUID
        Unique identifier for this canvas.
    scene_id : UUID
        ID of the scene this canvas renders.
    get_scene_fn : Callable[[UUID], gfx.Scene]
        Called each frame to retrieve the current scene.  Provided by
        ``RenderManager`` at construction time.
    dim : str
        Scene dimensionality: ``"2d"`` or ``"3d"``.  Controls which
        camera type and interaction controller are used.
    parent : QWidget or None
        Parent widget for the underlying ``QRenderWidget``.
    fov : float
        Vertical field of view in degrees (3D perspective only).
    depth_range : tuple[float, float]
        Near and far clip distances ``(near, far)``.
    """

    def __init__(
        self,
        canvas_id: UUID,
        scene_id: UUID,
        get_scene_fn: Callable[[UUID], gfx.Scene],
        dim: str = "3d",
        parent: QWidget | None = None,
        fov: float = 70.0,
        depth_range: tuple[float, float] = (1.0, 8000.0),
        event_bus: EventBus | None = None,
    ) -> None:
        self._canvas_id = canvas_id
        self._scene_id = scene_id
        self._get_scene_fn = get_scene_fn
        self._applying_model_state: bool = False
        self._id: UUID = uuid4()
        self._dim = dim
        self._event_bus: EventBus | None = event_bus
        self._camera_dirty: bool = False

        self._canvas = QRenderWidget(parent=parent, update_mode="continuous")
        self._renderer = gfx.WgpuRenderer(self._canvas)

        if dim == "2d":
            self._camera = gfx.OrthographicCamera(maintain_aspect=True)
            self._controller = gfx.PanZoomController(
                camera=self._camera, register_events=self._renderer
            )
        else:
            self._camera = gfx.PerspectiveCamera(fov, 16 / 9, depth_range=depth_range)
            self._controller = gfx.OrbitController(
                camera=self._camera, register_events=self._renderer
            )

        self._last_camera_state: CameraState = self._capture_camera_state()
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

        All array fields are copied.  Screen size is read from the canvas
        at call time and baked into the returned request.

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
        screen_w, screen_h = self._canvas.get_logical_size()

        if self._dim == "2d":
            return self._capture_orthographic(
                dims_state, target_visual_ids, screen_w, screen_h
            )
        return self._capture_perspective(
            dims_state, target_visual_ids, screen_w, screen_h
        )

    def _capture_perspective(
        self,
        dims_state: DimsState,
        target_visual_ids: frozenset[UUID] | None,
        screen_w: float,
        screen_h: float,
    ) -> ReslicingRequest:
        """Build a ReslicingRequest for a perspective camera."""
        return ReslicingRequest(
            camera_type="perspective",
            camera_pos=np.array(self._camera.world.position, dtype=np.float64),
            frustum_corners=np.asarray(self._camera.frustum, dtype=np.float64).copy(),
            fov_y_rad=float(np.radians(self._camera.fov)),
            screen_size_px=(float(screen_w), float(screen_h)),
            world_extent=(0.0, 0.0),
            dims_state=dims_state,
            request_id=uuid4(),
            scene_id=self._scene_id,
            target_visual_ids=target_visual_ids,
        )

    def _capture_orthographic(
        self,
        dims_state: DimsState,
        target_visual_ids: frozenset[UUID] | None,
        screen_w: float,
        screen_h: float,
    ) -> ReslicingRequest:
        """Build a ReslicingRequest for an orthographic camera.

        Computes the actual visible world extent accounting for the
        canvas aspect ratio.  The OrthographicCamera exposes ``width``
        and ``height`` which define the *minimum* visible extent; the
        actual extent is expanded in one dimension to match the canvas
        aspect ratio.
        """
        cam = self._camera
        vw = screen_w if screen_w > 0 else 800.0
        vh = screen_h if screen_h > 0 else 600.0
        canvas_aspect = vw / vh

        cam_w = cam.width if cam.width > 0 else 1.0
        cam_h = cam.height if cam.height > 0 else 1.0
        cam_aspect = cam_w / cam_h

        if canvas_aspect >= cam_aspect:
            world_height = cam_h
            world_width = cam_h * canvas_aspect
        else:
            world_width = cam_w
            world_height = cam_w / canvas_aspect

        return ReslicingRequest(
            camera_type="orthographic",
            camera_pos=np.array(cam.world.position, dtype=np.float64),
            frustum_corners=np.zeros((2, 4, 3), dtype=np.float64),
            fov_y_rad=0.0,
            screen_size_px=(float(vw), float(vh)),
            world_extent=(float(world_width), float(world_height)),
            dims_state=dims_state,
            request_id=uuid4(),
            scene_id=self._scene_id,
            target_visual_ids=target_visual_ids,
        )

    def set_depth_range(self, depth_range: tuple[float, float]) -> None:
        """Set the camera near/far clip distances.

        Parameters
        ----------
        depth_range : tuple[float, float]
            ``(near, far)`` clip distances in world units.
        """
        self._camera.depth_range = depth_range

    def show_object(self, scene: gfx.Scene) -> None:
        """Fit the camera to the scene bounding box.

        Parameters
        ----------
        scene : gfx.Scene
            The scene to fit the camera to.
        """
        if self._dim == "2d":
            self._camera.show_object(scene, view_dir=(0, 0, -1), up=(0, 1, 0))
        else:
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
        self._last_camera_state = self._capture_camera_state()

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Wire the EventBus after construction."""
        self._event_bus = event_bus

    def _capture_camera_state(self) -> CameraState:
        """Snapshot the current pygfx camera into a CameraState NamedTuple."""
        cam = self._camera
        pos = cam.world.position
        rot = cam.world.rotation  # quaternion (x, y, z, w)
        dr = cam.depth_range
        depth_range = (float(dr[0]), float(dr[1])) if dr is not None else (0.0, 0.0)

        if self._dim == "2d":
            return CameraState(
                camera_type="orthographic",
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                rotation=(float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])),
                up=(0.0, 1.0, 0.0),
                fov=0.0,
                zoom=float(cam.zoom),
                extent=(float(cam.width), float(cam.height)),
                depth_range=depth_range,
            )
        else:
            up = cam.world.up
            return CameraState(
                camera_type="perspective",
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                rotation=(float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])),
                up=(float(up[0]), float(up[1]), float(up[2])),
                fov=float(cam.fov),
                zoom=float(cam.zoom),
                extent=(0.0, 0.0),
                depth_range=depth_range,
            )

    def _draw_frame(self) -> None:
        # Detect camera changes by comparing against the cached state.
        current_state = self._capture_camera_state()
        if current_state != self._last_camera_state and not self._applying_model_state:
            self._camera_dirty = True
            self._last_camera_state = current_state
            _CAMERA_LOGGER.debug(
                "camera_changed  canvas=%s  scene=%s",
                self._canvas_id,
                self._scene_id,
            )

        if self._camera_dirty and self._event_bus is not None:
            self._camera_dirty = False
            _CAMERA_LOGGER.debug(
                "emit_camera_event  canvas=%s  scene=%s",
                self._canvas_id,
                self._scene_id,
            )
            self._event_bus.emit(
                CameraChangedEvent(
                    source_id=self._canvas_id,
                    scene_id=self._scene_id,
                    camera_state=current_state,
                )
            )

        scene = self._get_scene_fn(self._scene_id)
        self._renderer.render(scene, self._camera)
