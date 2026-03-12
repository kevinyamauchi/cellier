"""PyGFX render node for chunked 3D volume rendering.

:class:`GFXChunkedImageNode` wraps a
:class:`~cellier.render._texture_atlas.TextureAtlas` and a single
:class:`pygfx.Volume` that is lazily initialised on the first
:class:`~cellier.types.ChunkedDataResponse` the node receives.  The atlas
texture is attached to the volume's geometry; the
``texture_to_world_transform`` from each response is applied to the volume's
``local.matrix`` so it is correctly positioned in world space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import pygfx as gfx

from cellier.models.visuals.image import ImageAppearance
from cellier.render._texture_atlas import TextureAtlas

if TYPE_CHECKING:
    from cellier.models.visuals.chunked_image import ChunkedImageVisual
    from cellier.transform import BaseTransform
    from cellier.types import ChunkData, ChunkedDataResponse
    from cellier.utils.chunked_image._data_classes import TextureConfiguration


class GFXChunkedImageNode:
    """PyGFX render node for chunked 3D volume rendering.

    On construction only the parent :class:`pygfx.Group` and the
    :class:`pygfx.VolumeMipMaterial` are created.  The
    :class:`~cellier.render._texture_atlas.TextureAtlas` and
    :class:`pygfx.Volume` child node are created lazily on the first call to
    :meth:`set_slice`, using the
    :class:`~cellier.utils.chunked_image._data_classes.TextureConfiguration`
    embedded in the :class:`~cellier.types.ChunkedDataResponse`.

    Parameters
    ----------
    model : ChunkedImageVisual
        Visual model carrying appearance settings (colour map, visibility).
    """

    def __init__(self, model: ChunkedImageVisual) -> None:
        appearance_model = model.appearance
        if isinstance(appearance_model, ImageAppearance):
            pygfx_cm = appearance_model.color_map.to_pygfx(N=256)
            self._material = gfx.VolumeMipMaterial(map=pygfx_cm)
        else:
            raise TypeError(
                f"GFXChunkedImageNode requires ImageAppearance, "
                f"got {type(appearance_model).__name__}"
            )

        # Parent group — always present.
        self.node: gfx.Group = gfx.Group(
            name=model.id, visible=model.appearance.visible
        )

        # Lazily created on first set_slice().  One Volume per scale level;
        # only the active scale's Volume is visible at any given time.
        self._volume_nodes: list[gfx.Volume] = []
        self._active_scale: int = 0
        self._atlas: TextureAtlas | None = None

    # ------------------------------------------------------------------
    # Render-node interface expected by RenderManager / add_visual
    # ------------------------------------------------------------------

    @property
    def callback_handlers(self) -> list[Callable]:
        """Mouse-event handler registration callables for each sub-node."""
        return [vol.add_event_handler for vol in self._volume_nodes]

    def set_slice(self, slice_data: ChunkedDataResponse) -> None:  # type: ignore[override]
        """Apply a :class:`~cellier.types.ChunkedDataResponse` to the node.

        On the first call the texture atlas and all per-scale volume sub-nodes
        are created from ``slice_data.texture_config``.  On every call the
        active scale's world transform is updated from
        ``slice_data.texture_to_world_transform``, visibility is toggled so
        only the active scale is shown, and available chunks are uploaded to
        the atlas.

        Parameters
        ----------
        slice_data : ChunkedDataResponse
            Response produced by
            :class:`~cellier.models.data_stores.chunked_image.ChunkedImageStore`.
        """
        if self._atlas is None:
            self._initialize_atlas(slice_data.texture_config)

        active = slice_data.resolution_level

        # Position the active scale's volume in world space.
        self._volume_nodes[
            active
        ].local.matrix = slice_data.texture_to_world_transform.matrix

        # Show only the active scale.
        for i, vol in enumerate(self._volume_nodes):
            vol.visible = i == active

        self._active_scale = active

        # Upload chunks that are immediately available from the CPU cache.
        for chunk_data in slice_data.available_chunks:
            self._atlas.upload_chunk(chunk_data)  # type: ignore[union-attr]

    def upload_chunk(self, chunk_data: ChunkData) -> None:
        """Upload a single chunk that finished loading in the background.

        Called by the :class:`~cellier.render._render_manager.RenderManager`
        when ``ChunkManager.chunk_loaded``
        fires.  Silently drops the chunk if the atlas has not been
        initialised yet (e.g. the visual has not received its first slice).

        Parameters
        ----------
        chunk_data : ChunkData
            Chunk to upload.
        """
        if self._atlas is None:
            return
        self._atlas.upload_chunk(chunk_data)

    def update_appearance(self, new_state: dict) -> None:
        """Update visual appearance from a model-change event.

        Parameters
        ----------
        new_state : dict
            Subset of appearance fields that changed.
        """
        if "visible" in new_state:
            self.node.visible = new_state["visible"]

    def set_transform(self, transform: BaseTransform) -> None:
        """Set the local-space transform of the group node.

        Parameters
        ----------
        transform : BaseTransform
            The 4x4 affine transform to apply.
        """
        self.node.local.matrix = transform.matrix

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _initialize_atlas(self, texture_config: TextureConfiguration) -> None:
        """Create the TextureAtlas and one Volume sub-node per scale on first use.

        Parameters
        ----------
        texture_config : TextureConfiguration
            Specifies ``texture_width`` (edge length in voxels) and
            ``n_scales`` (number of scale levels to allocate).
        """
        tw = texture_config.texture_width
        n_scales = texture_config.n_scales
        self._atlas = TextureAtlas(texture_width=tw, n_scales=n_scales)

        for i in range(n_scales):
            texture = self._atlas.texture_for_scale(i)
            geometry = gfx.Geometry(grid=texture)
            volume = gfx.Volume(geometry=geometry, material=self._material)
            volume.visible = i == 0  # only scale 0 visible initially
            self.node.add(volume)
            self._volume_nodes.append(volume)
