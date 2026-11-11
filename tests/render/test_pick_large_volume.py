"""A 3D surface pick reads the right voxel on a long axis, not only a short one.

The pick buffer holds each axis in 14 bits, so the quantum is ``dim / 16383``
voxels of the longest axis (the in-memory shaders use the longest one for every
axis).  The B12 face tests use a 40-voxel volume, where the quantum is 0.002 of
a voxel.  Here the volume is 24 x 32 x ``L`` with ``L`` up to 16000, so the
quantum grows to 0.6 of a voxel while the volume stays cheap to hold.  The block
is placed off the middle, where its faces would sit on the quantisation grid.

Every pick on the 3-voxel block must read the block's value.  The shaders clamp
the reported point one quantum inside its voxel and round to the nearest step,
which holds for any axis shorter than 16383 voxels.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.data import ImageMemoryStore, LabelMemoryStore
from cellier.visuals import (
    InMemoryImageSingleAppearance,
    InMemoryLabelsAppearance,
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests.render._affine import bind, canvas_view, load, multiscale_store
from tests.render.test_labels_pick_face import _pick_every_view, _viewer

DISPLAYED = ("z", "y", "x")
FAMILIES = ("labels", "labels_multiscale", "image", "image_multiscale")


#: Room for every brick of the volume (250 bricks of 64 ** 3 voxels at the
#: longest).  The cache is a cube of slots, so the budget moves in big steps: 320
#: MiB is 215 usable slots, 512 MiB is 342.
def _budgets(length: int) -> dict[str, int]:
    mib = 320 if length <= 10000 else 512
    return {"gpu_budget_bytes": mib * 1024**2, "gpu_budget_bytes_2d": 16 * 1024**2}


#: A 3D texture is at most 2048 voxels on a side, which bounds the in-memory
#: families; a multiscale one is bounded by its LUT (255 bricks per axis, so
#: 16320 voxels at ``block_size=64``).  8192 is in on purpose, with 8000: the
#: shader has to hold wherever the faces fall on the quantisation grid.  Above
#: about 10500 the fitted camera's far plane (8000) leaves the block's bricks
#: unplanned; ``_run`` widens it and reloads (see ``_widen_far_plane``).
_MULTISCALE = [1024, 4096, 5000, 8000, 8192, 10000, 12000, 16000]
LENGTHS = {
    "labels": [1024, 2048],
    "image": [1024, 2048],
    "labels_multiscale": _MULTISCALE,
    "image_multiscale": _MULTISCALE,
}
_CASES = [(family, n) for family in FAMILIES for n in LENGTHS[family]]


def _volume(
    length: int, dtype, x_centre: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """A 3-voxel block of ones in a ``(24, 32, length)`` volume, and its centre."""
    # Not the middle: there the faces sit on the quantisation grid (multiples of
    # ``dim / 16383``) and the error vanishes by coincidence.
    if x_centre is None:
        x_centre = int(length * 0.4137)
    centre = np.array([7, 20, x_centre])
    data = np.zeros((24, 32, length), dtype=dtype)
    z, y, x = centre
    data[z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2] = 1
    return data, centre


def _add(viewer, family, data, tmp_path, render_mode=None):
    identity = np.eye(4)
    if family == "labels":
        store = LabelMemoryStore(data=data, name="long")
        return viewer.add_labels(
            store,
            appearance=InMemoryLabelsAppearance(render_mode="iso_categorical"),
            transform=bind(viewer, store, identity),
        )
    if family == "image":
        store = ImageMemoryStore(data=data, name="long")
        return viewer.add_image(
            store,
            single=InMemoryImageSingleAppearance(
                color_map="gray",
                clim=(0.0, 1.0),
                render_mode=render_mode or "iso",
                iso_threshold=0.5,
            ),
            transform=bind(viewer, store, identity),
        )
    store = multiscale_store(tmp_path, "long", data)
    if family == "labels_multiscale":
        return viewer.add_labels_multiscale(
            store,
            appearance=MultiscaleLabelsAppearance(
                force_level=0, ray_steps_per_voxel=4.0
            ),
            render_config=MultiscaleLabelRenderConfig(
                **_budgets(data.shape[-1]), block_size=64
            ),
            transform=bind(viewer, store, identity),
        )
    return viewer.add_image_multiscale(
        store,
        appearance=MultiscaleImageAppearance(force_level=0),
        render_config=MultiscaleImageRenderConfig(
            **_budgets(data.shape[-1]), block_size=64
        ),
        single=MultiscaleImageSingleAppearance(
            color_map="gray",
            clim=(0.0, 1.0),
            render_mode=render_mode or "iso",
            iso_threshold=0.5,
        ),
        transform=bind(viewer, store, identity),
    )


def _widen_far_plane(viewer, length: int) -> None:
    """Plan the whole volume, not only what the fitted camera's far plane reaches.

    The default 3D depth range is ``(1, 8000)`` and the planner culls bricks
    against the camera frustum.  The fitted camera of a volume longer than about
    10500 voxels is more than 8000 from its near end, so those bricks are never
    loaded, and the test camera, moved next to the block afterwards, sees an
    empty lookup table.  Setting the range before loading is not enough (the
    first load is what makes the camera), so the range is set and loaded again.
    """
    camera = canvas_view(viewer).camera
    camera.depth_range = (1.0, 4.0 * length)
    load(viewer)


def _run(tmp_path, family, length, x_centre=None, render_mode=None):
    viewer = _viewer(DISPLAYED)
    try:
        dtype = np.int32 if family.startswith("labels") else np.float32
        data, centre = _volume(length, dtype, x_centre)
        visual = _add(viewer, family, data, tmp_path, render_mode)
        load(viewer)
        _widen_far_plane(viewer, length)
        wrong = []

        def check(direction, level, full):
            voxel = tuple(int(v) for v in np.floor(np.asarray(full)))
            inside = all(0 <= v < s for v, s in zip(voxel, data.shape, strict=True))
            if not inside or data[voxel] != 1:
                wrong.append((direction.round(2).tolist(), full))

        n_picks = _pick_every_view(viewer, visual, DISPLAYED, check, centre=centre)
        assert not wrong, (
            f"{len(wrong)} of {n_picks} picks read background: {wrong[:3]}"
        )
    finally:
        viewer.controller.close()


@pytest.mark.parametrize(("family", "length"), _CASES)
def test_surface_picks_read_the_block_on_a_long_axis(tmp_path, family, length):
    _run(tmp_path, family, length)


#: The block's low face at ``64 k + offset``: from a voxel before a brick
#: boundary to three after it.  The ray used to skip a fixed ``1e-4`` of the
#: next brick, which is 0.8 of a voxel on an 8192-voxel axis, so a surface this
#: close to a boundary was found a voxel off (up to 9% of picks).
@pytest.mark.parametrize("offset", [-2, -1, 0, 1, 2, 3])
@pytest.mark.parametrize("length", [4096, 8192])
@pytest.mark.parametrize("family", ["labels_multiscale", "image_multiscale"])
def test_surface_picks_next_to_a_brick_boundary(tmp_path, family, length, offset):
    _run(tmp_path, family, length, (length // 128) * 64 + 64 + offset + 1)


#: A maximum intensity projection reports where the peak sample is: the first
#: sample that reaches the peak, which with nearest sampling sits at the face
#: the ray entered the block by.  The multiscale shader truncated that point
#: and read the neighbour (1% of picks at 1024 voxels, 30% at 10000).
_MIP_CASES = [
    ("image", "mip", 1024),
    ("image", "mip", 2048),
    *[
        (family, mode, n)
        for family, mode in (
            ("image_multiscale", "mip"),
            ("image_multiscale", "attenuated_mip"),
        )
        for n in (1024, 4096, 8192, 10000, 16000)
    ],
]


@pytest.mark.parametrize(("family", "mode", "length"), _MIP_CASES)
def test_mip_picks_read_the_block_on_a_long_axis(tmp_path, family, mode, length):
    _run(tmp_path, family, length, render_mode=mode)
