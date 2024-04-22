"""RenderManager class contains all of the rendering and visuals code."""

from typing import Dict, NamedTuple

import pygfx as gfx
from pygfx.renderers import WgpuRenderer
from wgpu.gui import WgpuCanvasBase

from cellier.models.viewer import ViewerModel
from cellier.render.utils import construct_pygfx_object


class VisualKey(NamedTuple):
    """The key to a visual stored in the RenderManager.

    Attributes
    ----------
    scene_id : str
        The uid of the scene model this visual belongs to.
    visual_id : str
        The uid of the visual model this pygfx belongs to.
    """

    scene_id: str
    visual_id: str


class RenderManager:
    """Class to manage the rendering."""

    def __init__(self, viewer_model: ViewerModel, canvases: Dict[str, WgpuCanvasBase]):
        # make each scene
        renderers = {}
        scenes = {}
        visuals = {}
        for scene_model in viewer_model.scenes.scenes:
            # make a renderer
            scene_id = scene_model.id
            renderers.update({scene_id: WgpuRenderer(canvases[scene_id])})

            # make a scene
            scene = gfx.Scene()
            scenes.update({scene_id: scene})

            # populate the scene
            for visual_model in scene_model.visuals:
                key = VisualKey(scene_id=scene_model.id, visual_id=visual_model.id)
                visuals.update({key: construct_pygfx_object(visual_model=visual_model)})

        # store the values
        self._renderers = renderers
        self._scenes = scenes
        self._visuals = visuals

    def renderers(self) -> Dict[str, WgpuRenderer]:
        """Dictionary of renderers.

        The key is the id property of the Scene object the renderer
        belongs to.
        """
        return self._renderers
