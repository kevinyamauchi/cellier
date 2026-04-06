"""Components for the rendering backend."""

from cellier.v2.render._config import (
    CameraConfig,
    RenderManagerConfig,
    SlicingConfig,
    TemporalAccumulationConfig,
)
from cellier.v2.render._requests import DimsState, ReslicingRequest
from cellier.v2.render._scene_config import VisualRenderConfig
from cellier.v2.render._temporal_accumulation import TemporalAccumulationPass
from cellier.v2.render.canvas_view import CanvasView
from cellier.v2.render.render_manager import RenderManager
from cellier.v2.render.scene_manager import SceneManager
from cellier.v2.render.slice_coordinator import SliceCoordinator

__all__ = [
    "CameraConfig",
    "CanvasView",
    "DimsState",
    "RenderManager",
    "RenderManagerConfig",
    "ReslicingRequest",
    "SceneManager",
    "SliceCoordinator",
    "SlicingConfig",
    "TemporalAccumulationConfig",
    "TemporalAccumulationPass",
    "VisualRenderConfig",
]
