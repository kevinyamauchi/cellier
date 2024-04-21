"""RenderManager class contains all of the rendering and visuals code."""

from typing import Dict

import pygfx as gfx
from pygfx.renderers import WgpuRenderer
from wgpu.gui import WgpuCanvasBase

from cellier.models.viewer import ViewerModel


class RenderManager:
    """Class to manage the rendering."""

    def __init__(self, viewer_model: ViewerModel, canvases: Dict[str, WgpuCanvasBase]):
        # make each scene
        renderers = {}
        scenes = {}
        for scene_model in viewer_model.scenes.scenes:
            # make a renderer
            scene_id = scene_model.id
            renderers.update({scene_id: WgpuRenderer(canvases[scene_id])})

            # make a scene
            scene = gfx.Scene()
            scenes.update({scene_id: scene})

        # store the values
        self._renderers = renderers
        self._scenes = scenes

    def renderers(self) -> Dict[str, WgpuRenderer]:
        """Dictionary of renderers.

        The key is the id property of the Scene object the renderer
        belongs to.
        """
        return self._renderers
