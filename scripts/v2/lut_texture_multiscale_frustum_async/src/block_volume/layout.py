"""Block layout parameters for bricked volume rendering.

Computes all derived layout constants from the volume shape and block size.
This is the Python equivalent of BVV's ``CacheSpec``.
"""

import math
from dataclasses import dataclass

BLOCK_SIZE_DEFAULT = 32


@dataclass(frozen=True)
class BlockLayout:
    """Derived layout constants for a bricked volume.

    Parameters
    ----------
    volume_shape : tuple[int, int, int]
        Volume dimensions in (D, H, W) numpy axis order.
    block_size : int
        Number of voxels per brick side (same in all three axes).

    Attributes
    ----------
    grid_dims : tuple[int, int, int]
        Number of bricks in each axis (gD, gH, gW).
        Also the LUT texture shape.
    padded_shape : tuple[int, int, int]
        Cache texture dimensions (pD, pH, pW) rounded up to full bricks.
    n_bricks : int
        Total number of bricks.
    """

    volume_shape: tuple[int, int, int]
    block_size: int
    grid_dims: tuple[int, int, int]
    padded_shape: tuple[int, int, int]
    n_bricks: int

    @classmethod
    def from_volume_shape(
        cls,
        volume_shape: tuple[int, int, int],
        block_size: int = BLOCK_SIZE_DEFAULT,
    ) -> "BlockLayout":
        """Create a BlockLayout from a volume shape.

        Parameters
        ----------
        volume_shape : tuple[int, int, int]
            Volume dimensions in (D, H, W) numpy axis order.
        block_size : int
            Number of voxels per brick side.

        Returns
        -------
        layout : BlockLayout
            The computed layout.

        Raises
        ------
        ValueError
            If any grid dimension exceeds 255 (uint8 LUT limit).
        """
        d, h, w = volume_shape
        gd = math.ceil(d / block_size)
        gh = math.ceil(h / block_size)
        gw = math.ceil(w / block_size)

        if max(gd, gh, gw) > 255:
            raise ValueError(
                f"Grid dimension {max(gd, gh, gw)} exceeds uint8 range (255). "
                "Reduce block_size or use a uint16 LUT format."
            )

        return cls(
            volume_shape=volume_shape,
            block_size=block_size,
            grid_dims=(gd, gh, gw),
            padded_shape=(gd * block_size, gh * block_size, gw * block_size),
            n_bricks=gd * gh * gw,
        )
