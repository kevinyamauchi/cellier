"""Utilities for working with the viewer model."""

from cellier.models.scene import Canvas, DimsManager, Scene
from cellier.models.viewer import ViewerModel
from cellier.types import CanvasId, DimsId, VisualId


def get_dims_with_visual_id(
    viewer_model: ViewerModel, visual_id: VisualId
) -> DimsManager:
    """Get the dims manager for a visual ID.

    Parameters
    ----------
    viewer_model : ViewerModel
        The viewer model.
    visual_id : VisualId
        The visual ID.

    Returns
    -------
    DimsManager
        The dims manager for the visual ID.
    """
    # get the scene model for the visual ID
    scenes = viewer_model.scenes.scenes

    for scene_model in scenes.values():
        for visual_model in scene_model.visuals:
            if visual_model.id == visual_id:
                return scene_model.dims
    raise ValueError(f"Visual ID {visual_id} not found in any scene.")


def get_dims_with_canvas_id(
    viewer_model: ViewerModel, canvas_id: CanvasId
) -> DimsManager:
    """Get the dims manager for a visual ID.

    Parameters
    ----------
    viewer_model : ViewerModel
        The viewer model.
    canvas_id : CanvasId
        The canvas ID.

    Returns
    -------
    DimsManager
        The dims manager for the visual ID.
    """
    # get the scene model for the visual ID
    scenes = viewer_model.scenes.scenes

    for scene_model in scenes.values():
        for canvas_model in scene_model.canvases.values():
            if canvas_model.id == canvas_id:
                return scene_model.dims
    raise ValueError(f"Canvas ID {canvas_id} not found in any scene.")


def get_canvas_with_visual_id(
    viewer_model: ViewerModel,
    visual_id: VisualId,
) -> Canvas:
    """Get the canvas model from the visual id.

    Note: this assumes that there is only one canvas per scene.
    """
    for scene in viewer_model.scenes.scenes.values():
        for visual in scene.visuals:
            if visual.id == visual_id:
                canvas_model = next(iter(scene.canvases.values()))
                return canvas_model
    raise ValueError(f"Visual id {visual_id} not found in any scene.")


def get_scene_with_dims_id(
    viewer_model: ViewerModel,
    dims_id: DimsId,
) -> Scene:
    """Get the scene model from the dims id.

    Parameters
    ----------
    viewer_model : ViewerModel
        The viewer model.
    dims_id : DimsId
        The dims id to search for the scene with..
    """
    for scene in viewer_model.scenes.scenes.values():
        if scene.dims.id == dims_id:
            return scene

    raise (ValueError(f"Dims id {dims_id} not found in any scene."))
