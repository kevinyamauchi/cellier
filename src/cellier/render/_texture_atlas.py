"""GPU texture atlas for chunked 3D volume rendering.

The atlas pre-allocates one cubic ``texture_width³`` GPU texture per scale
level.  Individual chunks are written into the correct sub-region of the
texture via ``send_data``.  The atlas tracks which ``(scale_index,
chunk_index)`` pairs are currently resident so callers can query membership
and evict stale chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pygfx as gfx

    from cellier.types import ChunkData


@dataclass
class TextureRegion:
    """Location of a single chunk within the texture atlas.

    Parameters
    ----------
    offset : tuple[int, int, int]
        ``(z, y, x)`` voxel offset of the chunk's min corner inside the
        atlas texture.
    shape : tuple[int, int, int]
        ``(z, y, x)`` voxel extent of the chunk.
    """

    offset: tuple[int, int, int]
    shape: tuple[int, int, int]


class TextureAtlas:
    """Pre-allocated GPU texture atlas for chunked 3D volume rendering.

    One cubic ``texture_width³`` :class:`pygfx.Texture` is created per scale
    level at construction time.  Individual chunks are uploaded to their
    correct sub-region via :meth:`upload_chunk`.

    Parameters
    ----------
    texture_width : int
        Edge length (in voxels) of each cubic texture.  All three dimensions
        are the same size.
    n_scales : int, optional
        Number of scale levels to allocate textures for.  Default is 1.

    Notes
    -----
    Coordinate convention: all offsets and shapes stored here and passed to
    :meth:`upload_chunk` use ``(z, y, x)`` ordering, consistent with the rest
    of Cellier.  Internally the ``(z, y, x)`` offset is reversed to
    ``(x, y, z)`` when calling :func:`pygfx.Texture.send_data`, which follows
    PyGFX's ``(x, y, z)`` convention.
    """

    def __init__(self, texture_width: int, n_scales: int = 1) -> None:
        import pygfx as gfx
        import wgpu

        self._texture_width = texture_width
        self._textures: list[gfx.Texture] = []

        for _ in range(n_scales):
            # Create without local data so send_data() can be used to upload
            # individual chunk sub-regions.  Textures created with data= store
            # a local backing array and pygfx refuses send_data() on them.
            texture = gfx.Texture(
                size=(texture_width, texture_width, texture_width),
                dim=3,
                format="1xf4",
                usage=wgpu.TextureUsage.COPY_DST,
            )
            self._textures.append(texture)

        # (scale_index, chunk_index) → TextureRegion
        self._regions: dict[tuple[int, int], TextureRegion] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def texture_width(self) -> int:
        """Edge length of each atlas texture in voxels."""
        return self._texture_width

    def texture_for_scale(self, scale_index: int) -> gfx.Texture:
        """Return the pre-allocated GPU texture for *scale_index*.

        Parameters
        ----------
        scale_index : int
            Scale level whose texture is requested.

        Returns
        -------
        pygfx.Texture
            The texture object; pass to ``gfx.Geometry(grid=...)`` to
            attach it to a volume node.
        """
        return self._textures[scale_index]

    def upload_chunk(self, chunk_data: ChunkData) -> None:
        """Write a chunk's voxel data into the atlas texture.

        The destination is determined by ``chunk_data.texture_offset`` (in
        ``(z, y, x)`` voxel coordinates).  The offset is reversed to
        ``(x, y, z)`` before calling :func:`~pygfx.Texture.send_data`.

        Parameters
        ----------
        chunk_data : ChunkData
            Chunk to upload.  ``chunk_data.scale_index`` selects the texture;
            ``chunk_data.texture_offset`` gives the ``(z, y, x)`` voxel
            position within that texture; ``chunk_data.data`` is the 3-D
            ``float32`` array.
        """
        texture = self._textures[chunk_data.scale_index]
        offset_zyx = chunk_data.texture_offset  # (z, y, x) — Cellier convention

        # PyGFX send_data expects (x, y, z) ordering.
        offset_xyz = (offset_zyx[2], offset_zyx[1], offset_zyx[0])
        texture.send_data(offset_xyz, chunk_data.data)

        self._regions[(chunk_data.scale_index, chunk_data.chunk_index)] = TextureRegion(
            offset=offset_zyx,
            shape=chunk_data.data.shape,
        )

    def has_chunk(self, scale_index: int, chunk_index: int) -> bool:
        """Return True if the chunk is currently resident in the atlas.

        Parameters
        ----------
        scale_index : int
            Scale level to query.
        chunk_index : int
            Linear chunk index within that scale level.
        """
        return (scale_index, chunk_index) in self._regions

    def evict_chunk(self, scale_index: int, chunk_index: int) -> None:
        """Remove a chunk's residency record (does not zero GPU memory).

        Parameters
        ----------
        scale_index : int
            Scale level of the chunk to evict.
        chunk_index : int
            Linear chunk index within that scale level.
        """
        self._regions.pop((scale_index, chunk_index), None)

    def clear_scale(self, scale_index: int) -> None:
        """Zero every resident chunk region for *scale_index* on the GPU.

        Only the sub-regions that were previously written via
        :meth:`upload_chunk` are zeroed, rather than the entire
        ``texture_width³`` cube.  This keeps the GPU transfer cost
        proportional to the number of uploaded chunks rather than the
        full texture allocation.

        The residency records for *scale_index* are removed so that
        subsequent :meth:`has_chunk` queries return ``False``.

        Parameters
        ----------
        scale_index : int
            Scale level whose texture should be cleared.
        """
        import numpy as np

        texture = self._textures[scale_index]
        keys = [k for k in self._regions if k[0] == scale_index]
        for key in keys:
            region = self._regions.pop(key)
            zeros = np.zeros(region.shape, dtype=np.float32)
            # send_data expects (x, y, z); region.offset is (z, y, x).
            offset_xyz = (region.offset[2], region.offset[1], region.offset[0])
            texture.send_data(offset_xyz, zeros)
