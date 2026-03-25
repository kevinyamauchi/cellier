import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BlockLayout2D:
    """Derived layout constants for a tiled 2D image.

    Attributes
    ----------
    volume_shape : tuple[int, int]
        Image dimensions ``(H, W)`` in numpy axis order.
    block_size : int
        Number of pixels per tile side.
    overlap : int
        Border pixels duplicated on each side of every tile.
    grid_dims : tuple[int, int]
        Number of tiles per axis ``(gH, gW)``.
        Also the LUT texture shape.
    padded_shape : tuple[int, int]
        Image extent rounded up to full tiles plus overlap
        ``(pH, pW)`` where ``pH = gH * block_size + 2*overlap``.
    n_tiles : int
        Total number of tiles ``gH * gW``.
    """

    volume_shape: tuple[int, int]
    block_size: int
    overlap: int
    grid_dims: tuple[int, int]
    padded_shape: tuple[int, int]
    n_tiles: int

    @classmethod
    def from_shape(
        cls,
        shape: tuple[int, int],
        block_size: int = 32,
        overlap: int = 1,
    ) -> "BlockLayout2D":
        """Create a BlockLayout2D from an image shape.

        Parameters
        ----------
        shape : tuple[int, int]
            Image dimensions ``(H, W)`` in numpy axis order.
        block_size : int
            Number of pixels per tile side.
        overlap : int
            Border pixels duplicated on each side.

        Returns
        -------
        layout : BlockLayout2D
        """
        h, w = shape
        gh = math.ceil(h / block_size)
        gw = math.ceil(w / block_size)

        # padded_shape includes overlap border around the entire grid.
        ph = gh * block_size + 2 * overlap
        pw = gw * block_size + 2 * overlap

        return cls(
            volume_shape=(h, w),
            block_size=block_size,
            overlap=overlap,
            grid_dims=(gh, gw),
            padded_shape=(ph, pw),
            n_tiles=gh * gw,
        )
