"""Tests for cellier.transform_v2._affine (design section 9)."""

from uuid import uuid4

import numpy as np
import pytest

from cellier.transform_v2 import (
    AffineTransform,
    Axis,
    AxisAlignedBoundingBox,
    ConvexRegion,
    CoordinateSystem,
    DataCoordinateSystem,
    DegenerateNormalError,
    HalfSpace,
    NonInvertibleTransformError,
    Plane,
    RenderedCoordinateSystem,
    VisualCoordinateSystem,
    WorldCoordinateSystem,
)
from tests.transform_v2._use_cases import MODEL_USE_CASES, uc1, uc2, uc3, uc4

CASE_IDS = list(MODEL_USE_CASES)


def space(name):
    return Axis(name=name, axis_type="space", unit="micrometer")


def flat_2d(name="flat"):
    return CoordinateSystem(name=name, axes=(space("v"), space("u")))


def reducing_transform():
    """A 3D -> 2D projection: D24 case 3, inverse() is None."""
    data = CoordinateSystem(name="data", axes=(space("z"), space("y"), space("x")))
    flat = flat_2d()
    matrix = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    return data, flat, AffineTransform.from_matrix(matrix, data, flat)


# --- the four use cases (section 9.3) --------------------------------


def test_uc1_matrix_is_diagonal():
    _, _, transform = uc1()
    assert np.allclose(transform.matrix, np.diag([0.5, 0.1, 0.1, 1.0]))
    point = np.array([1.0, 2.0, 3.0])
    assert np.allclose(transform.map_coordinates(point), [0.5, 0.2, 0.3])


def test_uc4_matrix_is_anti_diagonal_and_the_call_is_identical_to_uc1():
    """The regression test for the axis-order bug class (section 3.1)."""
    _, _, one = uc1()
    _, _, four = uc4()
    assert np.allclose(four.matrix[:3, :3], [[0, 0, 0.1], [0, 0.1, 0], [0.5, 0, 0]])
    # same input point, different world axis order
    assert np.allclose(one.map_coordinates(np.array([1.0, 2.0, 3.0])), [0.5, 0.2, 0.3])
    assert np.allclose(four.map_coordinates(np.array([1.0, 2.0, 3.0])), [0.3, 0.2, 0.5])


def test_uc1_and_uc4_differ_only_in_the_world_system_d22():
    """The axis map, scale and every other argument are byte-identical."""
    import inspect

    from tests.transform_v2 import _use_cases

    one = inspect.getsource(_use_cases.uc1)
    four = inspect.getsource(_use_cases.uc4)
    call = (
        'axis_map={"z": "Z", "y": "Y", "x": "X"},\n'
        '        scale={"z": 0.5, "y": 0.1, "x": 0.1},'
    )
    assert call in one
    assert call in four


def test_uc2_channel_row_passes_through():
    _, _, transform = uc2()
    assert np.allclose(transform.matrix[0], [1.0, 0.0, 0.0, 0.0, 0.0])
    first = transform.map_coordinates(np.array([3.0, 1.0, 2.0, 3.0]))
    second = transform.map_coordinates(np.array([7.0, 1.0, 2.0, 3.0]))
    assert first[0] == 3.0 and second[0] == 7.0
    assert np.allclose(first[1:], second[1:])


def test_uc3_shape_rank_and_zero_channel_row():
    _, _, transform = uc3()
    assert transform.matrix.shape == (6, 5)
    assert transform.input_ndim == 4
    assert transform.output_ndim == 5
    assert np.allclose(transform.matrix[1], 0.0)
    assert transform.map_coordinates(np.array([1.0, 2.0, 3.0, 4.0]))[1] == 0.0


def test_uc3_inverse_exists_and_has_the_projecting_shape():
    _, _, transform = uc3()
    inverse = transform.inverse()
    assert inverse is not None
    assert inverse.matrix.shape == (5, 6)


def test_uc3_round_trip_is_exact():
    _, _, transform = uc3()
    points = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, 7.0]])
    assert np.allclose(
        transform.imap_coordinates(transform.map_coordinates(points)), points
    )


def test_uc3_perturbing_the_broadcast_axis_does_not_move_the_data_point():
    """The broadcast property, and the assertion that matters."""
    _, _, transform = uc3()
    point = np.array([1.0, 2.0, 3.0, 4.0])
    world = transform.map_coordinates(point)
    perturbed = world.copy()
    perturbed[1] = 99.0
    assert np.allclose(transform.imap_coordinates(perturbed), point)
    assert np.allclose(transform.imap_coordinates(world), point)


def test_uc3_records_the_broadcast_axis_d25():
    _, world, transform = uc3()
    assert transform.broadcast_axes == frozenset({world.axis_by_name("C").id})


def test_uc3_broadcast_axes_survives_a_round_trip():
    _, _, transform = uc3()
    back = AffineTransform.model_validate(transform.model_dump())
    assert back.broadcast_axes == transform.broadcast_axes


def test_uc3_without_broadcast_output_axes_raises_d22():
    data, world, _ = uc3()
    with pytest.raises(ValueError, match="neither mapped nor declared"):
        AffineTransform.from_axis_map(
            data,
            world,
            axis_map={"t": "T", "z": "Z", "y": "Y", "x": "X"},
            scale={"z": 0.5, "y": 0.1, "x": 0.1},
        )


@pytest.mark.parametrize("name", CASE_IDS)
def test_every_use_case_round_trips_points(name):
    _, _, transform = MODEL_USE_CASES[name]()
    rng = np.random.default_rng(0)
    points = rng.normal(size=(5, transform.input_ndim))
    assert np.allclose(
        transform.imap_coordinates(transform.map_coordinates(points)), points
    )


# --- points (D11, D12) -----------------------------------------------


def test_identity_affine_returns_a_new_array_d11():
    data = CoordinateSystem(name="d", axes=(space("z"), space("y")))
    world = CoordinateSystem(name="w", axes=(space("Z"), space("Y")))
    transform = AffineTransform.from_matrix(np.eye(3), data, world)
    points = np.array([[1.0, 2.0]])
    result = transform.map_coordinates(points)
    assert result is not points
    result[0, 0] = 99.0
    assert points[0, 0] == 1.0


def test_input_is_not_mutated():
    _, _, transform = uc2()
    points = np.array([[1.0, 2.0, 3.0, 4.0]])
    original = points.copy()
    transform.map_coordinates(points)
    assert np.array_equal(points, original)


def test_rank_of_output_matches_input_d12():
    _, _, transform = uc3()
    assert transform.map_coordinates(np.zeros(4)).shape == (5,)
    assert transform.map_coordinates(np.zeros((6, 4))).shape == (6, 5)
    assert transform.imap_coordinates(np.zeros(5)).shape == (4,)


def test_homogeneous_input_is_rejected_d12():
    _, _, transform = uc3()
    with pytest.raises(ValueError, match="Homogeneous"):
        transform.map_coordinates(np.zeros((3, 5)))


# --- inversion (D13, D24) ---------------------------------------------


def test_inverse_swaps_the_coordinate_system_ids_d13():
    """The explicit regression test for the Spaced.invert() bug."""
    data, world, transform = uc1()
    inverse = transform.inverse()
    assert inverse.input_coordinate_system == world.id
    assert inverse.output_coordinate_system == data.id
    assert transform.input_coordinate_system == data.id


def test_dimension_reducing_transform_has_no_inverse_d24_case_3():
    _, _, transform = reducing_transform()
    assert transform.input_ndim == 3
    assert transform.output_ndim == 2
    assert transform.map_coordinates(np.zeros((4, 3))).shape == (4, 2)
    assert transform.inverse() is None
    with pytest.raises(NonInvertibleTransformError):
        transform.imap_coordinates(np.zeros((4, 2)))


def test_singular_square_matrix_has_no_inverse():
    data = CoordinateSystem(name="d", axes=(space("z"), space("y")))
    world = CoordinateSystem(name="w", axes=(space("Z"), space("Y")))
    singular = np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
    transform = AffineTransform.from_matrix(singular, data, world)
    assert transform.inverse() is None


def test_inverse_is_cached():
    _, _, transform = uc3()
    assert transform._inverse_blocks is transform._inverse_blocks


def test_inverse_of_a_broadcast_transform_carries_no_broadcast_axes():
    _, _, transform = uc3()
    assert transform.inverse().broadcast_axes == frozenset()


# --- vectors (D27, D28) -----------------------------------------------


@pytest.mark.parametrize("name", CASE_IDS)
def test_normals_use_the_covariant_rule_d27(name):
    _, _, transform = MODEL_USE_CASES[name]()
    rng = np.random.default_rng(1)
    normal = rng.normal(size=transform.input_ndim)
    direction = rng.normal(size=transform.input_ndim)
    assert np.allclose(
        transform.map_normal(normal) @ transform.map_direction(direction),
        normal @ direction,
    )


def test_there_is_no_map_vector_d27():
    assert not hasattr(AffineTransform, "map_vector")
    assert not hasattr(AffineTransform, "imap_vector")


def test_nothing_is_normalised_d28():
    """v1 returned unit normals; v2 returns the raw transformed vector."""
    _, _, transform = uc1()  # scale (0.5, 0.1, 0.1)
    vector = np.array([3.0, 0.0, 0.0])
    # a displacement scales by A, and its magnitude is meaningful
    assert np.allclose(transform.map_direction(vector), [1.5, 0.0, 0.0])
    # a normal scales by (A+)^T, the other way
    assert np.allclose(transform.map_normal(vector), [6.0, 0.0, 0.0])
    assert np.allclose(transform.imap_normal(vector), [1.5, 0.0, 0.0])
    assert np.allclose(transform.imap_direction(vector), [6.0, 0.0, 0.0])


def test_map_direction_and_imap_normal_need_no_inverse_d27():
    _, _, transform = reducing_transform()
    assert transform.inverse() is None
    assert transform.map_direction(np.array([1.0, 2.0, 3.0])).shape == (2,)
    assert transform.imap_normal(np.array([1.0, 2.0])).shape == (3,)
    with pytest.raises(NonInvertibleTransformError):
        transform.map_normal(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(NonInvertibleTransformError):
        transform.imap_direction(np.array([1.0, 2.0]))


# --- planes (D29, D32) ------------------------------------------------


def test_map_plane_keeps_points_on_the_plane():
    data, world, transform = uc2()
    rng = np.random.default_rng(2)
    normal = rng.normal(size=4)
    offset = 1.5
    plane = Plane(coordinate_system=data.id, normal=normal, offset=offset)
    mapped = transform.map_plane(plane)
    assert mapped.coordinate_system == world.id

    sample = rng.normal(size=(8, 4))
    on_plane = (
        sample + ((offset - sample @ normal) / (normal @ normal))[:, None] * normal
    )
    image = transform.map_coordinates(on_plane)
    assert np.allclose(image @ mapped.normal, mapped.offset)


def test_a_world_space_plane_fed_to_map_plane_raises_d29():
    data, world, transform = uc1()
    plane = Plane(coordinate_system=world.id, normal=[0.0, 0.0, 1.0], offset=1.0)
    with pytest.raises(ValueError, match="this transform's input"):
        transform.map_plane(plane)
    data_plane = Plane(coordinate_system=data.id, normal=[0.0, 0.0, 1.0], offset=1.0)
    with pytest.raises(ValueError, match="this transform's output"):
        transform.imap_plane(data_plane)


def test_a_plane_parallel_to_the_broadcast_axis_raises_d32():
    _, world, transform = uc3()
    plane = Plane(
        coordinate_system=world.id, normal=[0.0, 1.0, 0.0, 0.0, 0.0], offset=2.0
    )
    with pytest.raises(DegenerateNormalError):
        transform.imap_plane(plane)
    with pytest.raises(DegenerateNormalError):
        transform.imap_normal(np.array([0.0, 1.0, 0.0, 0.0, 0.0]))


def test_a_plane_tilted_in_c_and_x_does_not_raise_d32():
    """Pinned so the non-injectivity is not later 'fixed' into an exception."""
    _, world, transform = uc3()
    plane = Plane(
        coordinate_system=world.id, normal=[0.0, 1.0, 0.0, 0.0, 1.0], offset=0.0
    )
    pulled = transform.imap_plane(plane)
    assert np.allclose(pulled.normal, [0.0, 0.0, 0.0, 0.1])


# --- bounding boxes (D30, D31) ----------------------------------------


def test_map_bounding_box_marks_the_broadcast_axis_unbounded_d31():
    data, world, transform = uc3()
    box = AxisAlignedBoundingBox(
        coordinate_system=data.id,
        min_coordinate=[0.0, 0.0, 0.0, 0.0],
        max_coordinate=[10.0, 20.0, 30.0, 40.0],
    )
    mapped = transform.map_bounding_box(box, world)
    assert mapped.min_coordinate[1] == -np.inf
    assert mapped.max_coordinate[1] == np.inf
    assert np.allclose(mapped.min_coordinate[[0, 2, 3, 4]], [0.0, 10.0, -2.0, 3.0])
    assert np.allclose(mapped.max_coordinate[[0, 2, 3, 4]], [10.0, 20.0, 1.0, 7.0])


def test_map_bounding_box_requires_the_matching_output_system():
    data, _, transform = uc3()
    box = AxisAlignedBoundingBox(
        coordinate_system=data.id,
        min_coordinate=np.zeros(4),
        max_coordinate=np.ones(4),
    )
    other = WorldCoordinateSystem(
        axes=(space("T"), space("C"), space("Z"), space("Y"), space("X"))
    )
    with pytest.raises(ValueError, match="output_coordinate_system must be"):
        transform.map_bounding_box(box, other)


def test_map_bounding_box_rejects_a_world_space_box_d29():
    _, world, transform = uc1()
    box = AxisAlignedBoundingBox(
        coordinate_system=world.id,
        min_coordinate=np.zeros(3),
        max_coordinate=np.ones(3),
    )
    with pytest.raises(ValueError, match="this transform's input"):
        transform.map_bounding_box(box, world)


@pytest.mark.parametrize("name", ["UC1", "UC2", "UC4"])
def test_map_bounding_box_is_exact_for_the_diagonal_cases_d30(name):
    data, world, transform = MODEL_USE_CASES[name]()
    ndim = transform.input_ndim
    lower = np.zeros(ndim)
    upper = np.arange(1.0, ndim + 1.0)
    box = AxisAlignedBoundingBox(
        coordinate_system=data.id, min_coordinate=lower, max_coordinate=upper
    )
    mapped = transform.map_bounding_box(box, world)

    corners = np.array(
        [
            [upper[i] if (k >> i) & 1 else lower[i] for i in range(ndim)]
            for k in range(2**ndim)
        ]
    )
    image = transform.map_coordinates(corners)
    assert np.allclose(mapped.min_coordinate, image.min(0))
    assert np.allclose(mapped.max_coordinate, image.max(0))


@pytest.mark.parametrize("extent", [1.0, np.inf])
def test_imap_bounding_box_drops_the_broadcast_axis_either_way_d31(extent):
    data, world, transform = uc3()
    box = AxisAlignedBoundingBox(
        coordinate_system=world.id,
        min_coordinate=[0.0, -extent, 0.0, 0.0, 0.0],
        max_coordinate=[1.0, extent, 1.0, 1.0, 1.0],
    )
    pulled = transform.imap_bounding_box(box)
    assert pulled.coordinate_system == data.id
    assert np.all(np.isfinite(pulled.min_coordinate))
    assert np.all(np.isfinite(pulled.max_coordinate))


# --- regions (D39, D41) -----------------------------------------------


def test_imap_region_returns_a_region_not_a_box_d41():
    data, world, transform = uc3()
    region = ConvexRegion.from_axis_slabs(world, {"T": (7.0, 0.5)})
    pulled = transform.imap_region(region, world)
    assert isinstance(pulled, ConvexRegion)
    assert pulled.coordinate_system == data.id
    assert pulled.ndim == 4


def test_imap_region_works_without_an_inverse_d39():
    data, flat, transform = reducing_transform()
    assert transform.inverse() is None
    region = ConvexRegion.from_axis_slabs(flat, {"v": (1.0, 0.5)})
    pulled = transform.imap_region(region, flat)
    assert pulled.ndim == 3
    with pytest.raises(NonInvertibleTransformError):
        transform.map_region(
            ConvexRegion.unbounded(
                CoordinateSystem(name="d", id=data.id, axes=data.axes)
            )
        )


def test_imap_region_agrees_with_the_direct_predicate_d39():
    """Without broadcast axes the pull-back is exactly the forward predicate.

    This is D39's original property, and D8 must not disturb it: for a
    transform with no broadcast axes the drop rule removes nothing, so
    ``imap_region`` is still raw ``A^T``.  UC2 is used rather than UC3
    because a *broadcast* axis deliberately breaks this equivalence --
    the forward map puts the data at ``C = 0``, which is not where it
    is, and pinning that is
    :func:`test_imap_region_drops_constraints_on_a_broadcast_axis_d8`.
    """
    _, world, transform = uc2()
    rng = np.random.default_rng(3)
    region = ConvexRegion(
        coordinate_system=world.id,
        ndim=4,
        half_spaces=tuple(
            HalfSpace(normal=normal, offset=float(offset))
            for normal, offset in zip(
                rng.normal(size=(4, 4)), rng.normal(size=4) * 3, strict=True
            )
        ),
    )
    pulled = transform.imap_region(region, world)
    points = rng.normal(size=(3000, 4)) * 4
    assert np.array_equal(
        pulled.contains(points),
        region.contains(transform.map_coordinates(points)),
    )


# --- D8: imap_region consults broadcast_axes --------------------------


def _exists_s_oracle(region, transform, broadcast_index, points):
    """Exact membership by brute force over the free (broadcast) axis.

    A point ``p`` is visible in ``region`` when **some** position ``s``
    on the broadcast axis puts it inside:

    ``EXISTS s :  (N M) p + N t + N[:, b] s <= o``

    Independent of :meth:`imap_region` -- it evaluates the world-space
    constraints directly rather than pulling anything back -- which is
    what makes it an oracle rather than a restatement.
    """
    normals = region.normals
    offsets = region.offsets
    base = points @ (normals @ transform.linear).T + normals @ transform.translation
    column = normals[:, broadcast_index]
    inside = np.zeros(len(points), dtype=bool)
    for s in np.linspace(-50.0, 50.0, 401):
        inside |= np.all(base + column * s <= offsets + 1e-9, axis=1)
    return inside


def _uc3_oblique_through_c(world):
    """A slab whose normal tilts through the broadcast axis C and Z."""
    normal = np.array([0.0, 0.7, 0.7, 0.0, 0.0])
    return ConvexRegion.from_plane_slab(world, normal, 21.9, 1.0)


def test_imap_region_drops_constraints_on_a_broadcast_axis_d8():
    """The failure D8 exists to fix, on the case the viewer hits first.

    Selecting a channel the dataset has no axis for must not empty the
    region: the dataset is declared broadcast over ``C``, so it exists at
    every ``C``.  Without the drop rule the ``C`` constraints pull back
    through the all-zero row to ``0 <= -1.5`` and the visual vanishes the
    moment the channel slider leaves zero.
    """
    data, world, transform = uc3()
    region = ConvexRegion.from_axis_slabs(world, {"T": (7.0, 0.5), "C": (2.0, 0.5)})
    assert len(region.half_spaces) == 4

    pulled = transform.imap_region(region, world)

    assert not pulled.is_empty()
    # only the two T constraints survive; both C constraints were dropped
    # before A^T rather than pulled back to a zero normal.
    assert len(pulled.half_spaces) == 2
    assert np.all(np.any(pulled.normals != 0.0, axis=1))
    assert pulled.coordinate_system == data.id

    box = pulled.bounding_box()
    assert box.min_coordinate[0] == pytest.approx(6.5)
    assert box.max_coordinate[0] == pytest.approx(7.5)


def test_imap_region_without_d8_would_have_been_empty():
    """Pin the old answer, so the fix cannot silently regress."""
    _, world, transform = uc3()
    region = ConvexRegion.from_axis_slabs(world, {"T": (7.0, 0.5), "C": (2.0, 0.5)})
    # the shipped-before-D8 arithmetic: raw A^T with nothing dropped
    unamended = transform._imap_region(region, ())
    assert unamended.is_empty()
    assert transform.imap_region(region, world).is_empty() is False


@pytest.mark.parametrize("kind", ["axis_aligned", "oblique_through_c"])
def test_the_d8_drop_rule_matches_an_exists_s_oracle(kind):
    """The measurement that justifies shipping the cheap rule.

    For an axis-aligned selection the Fourier-Motzkin elimination of the
    broadcast axis degenerates exactly to "drop every constraint whose
    normal touches it", because the single ``(+C, -C)`` pair combines to
    ``0 <= 2 * half_thickness``.  The same holds for a lone oblique slab
    tilted through the axis.
    """
    _, world, transform = uc3()
    broadcast_index = world.resolve("C")

    if kind == "axis_aligned":
        region = ConvexRegion.from_axis_slabs(world, {"T": (7.0, 0.5), "C": (1.5, 1.0)})
    else:
        region = _uc3_oblique_through_c(world)

    rng = np.random.default_rng(0)
    points = rng.uniform(-6.0, 12.0, size=(3000, 4))

    truth = _exists_s_oracle(region, transform, broadcast_index, points)
    drop_rule = transform.imap_region(region, world).contains(points)

    assert truth.any(), "the oracle must select something for this to mean anything"
    assert np.array_equal(drop_rule, truth)


def test_the_d8_drop_rule_diverges_only_where_the_design_says_it_does():
    """The documented limit, pinned as a limit rather than as a bug.

    An oblique slab tilted *through* the broadcast axis, intersected
    with a bound on that same axis, is the one shape where the pairwise
    Fourier-Motzkin combination produces a constraint the drop rule
    discards.  The rule is then **over**-inclusive -- it fetches more
    than it needs to, never less -- and no shipping selection has this
    shape.  When one does, this test is the place the generalisation
    lands.
    """
    _, world, transform = uc3()
    broadcast_index = world.resolve("C")
    region = ConvexRegion.intersection(
        _uc3_oblique_through_c(world),
        ConvexRegion.from_axis_slabs(world, {"C": (2.0, 0.5)}),
    )

    rng = np.random.default_rng(0)
    points = rng.uniform(-6.0, 12.0, size=(3000, 4))

    truth = _exists_s_oracle(region, transform, broadcast_index, points)
    drop_rule = transform.imap_region(region, world).contains(points)

    assert not np.array_equal(drop_rule, truth)
    # over-inclusive, never under-inclusive: every true point is kept
    assert np.all(drop_rule[truth])
    assert drop_rule.sum() > truth.sum()


@pytest.mark.parametrize("name", ["UC1", "UC2", "UC4"])
def test_d8_leaves_a_transform_without_broadcast_axes_bit_identical(name):
    """No broadcast axes means nothing to drop, so the rule is raw A^T.

    Checked against the closed form written out inline rather than
    against the implementation, so this is an independent statement of
    what the answer must be.
    """
    _, world, transform = MODEL_USE_CASES[name]()
    assert transform.broadcast_axes == frozenset()

    rng = np.random.default_rng(11)
    ndim = world.ndim
    region = ConvexRegion(
        coordinate_system=world.id,
        ndim=ndim,
        half_spaces=tuple(
            HalfSpace(normal=normal, offset=float(offset))
            for normal, offset in zip(
                rng.normal(size=(5, ndim)), rng.normal(size=5) * 3, strict=True
            )
        ),
    )
    pulled = transform.imap_region(region, world)

    assert np.allclose(pulled.normals, region.normals @ transform.linear)
    assert np.allclose(
        pulled.offsets, region.offsets - region.normals @ transform.translation
    )
    assert len(pulled.half_spaces) == len(region.half_spaces)


def test_a_constant_output_axis_is_not_a_broadcast_one_under_imap_region_d39():
    """D8 must not erase D39's vacuous / infeasible distinction.

    A *constant* zero row means "the source sits at exactly this
    position", so a selection elsewhere on that axis genuinely selects
    nothing.  Only a *broadcast* zero row means "the source is at every
    position".  The two look identical in the matrix, which is why
    ``broadcast_axes`` is stored beside it (D25).
    """
    data, world, _ = uc3()
    constant = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={"t": "T", "z": "Z", "y": "Y", "x": "X"},
        constant_output_axes={"C": 0.0},
    )
    assert constant.broadcast_axes == frozenset()

    on_the_constant = ConvexRegion.from_axis_slabs(world, {"C": (0.0, 0.5)})
    assert not constant.imap_region(on_the_constant, world).is_empty()

    off_the_constant = ConvexRegion.from_axis_slabs(world, {"C": (2.0, 0.5)})
    assert constant.imap_region(off_the_constant, world).is_empty()


def test_imap_region_requires_the_matching_output_system():
    _, world, transform = uc3()
    region = ConvexRegion.from_axis_slabs(world, {"T": (7.0, 0.5)})
    with pytest.raises(ValueError, match="output_coordinate_system must be"):
        transform.imap_region(region, tczyx_world())


def test_map_region_is_not_given_the_d8_treatment_d31():
    """The asymmetry is deliberate: broadcast is unboundedness forward.

    ``map_region`` keeps its single-argument signature.  Forward,
    ``broadcast_axes`` shows up as an unbounded extent
    (:meth:`map_bounding_box`), not as a dropped constraint, and D31
    draws that line.
    """
    import inspect

    forward = inspect.signature(AffineTransform.map_region).parameters
    backward = inspect.signature(AffineTransform.imap_region).parameters
    assert list(forward) == ["self", "region"]
    assert list(backward) == ["self", "region", "output_coordinate_system"]


# --- composition (D10, D19) -------------------------------------------


def test_then_composes_in_application_order_d10():
    data, world, first = uc1()
    second_world = WorldCoordinateSystem(
        name="world2", axes=(space("Z2"), space("Y2"), space("X2"))
    )
    second = AffineTransform.from_axis_map(
        world,
        second_world,
        axis_map={"Z": "Z2", "Y": "Y2", "X": "X2"},
        scale={"Z": 2.0, "Y": 2.0, "X": 2.0},
        translation={"Z": 1.0},
    )
    chained = first.then(second, world, second_world)
    point = np.array([1.0, 2.0, 3.0])
    assert np.allclose(
        chained.map_coordinates(point),
        second.map_coordinates(first.map_coordinates(point)),
    )
    assert chained.input_coordinate_system == data.id
    assert chained.output_coordinate_system == second_world.id


def test_then_raises_across_a_coordinate_system_mismatch_d19():
    _, world, first = uc1()
    other = WorldCoordinateSystem(
        name="other", axes=(space("Z"), space("Y"), space("X"))
    )
    second = AffineTransform.from_axis_map(
        other,
        other,
        axis_map={"Z": "Z", "Y": "Y", "X": "X"},
    )
    # the ranks agree, and it still raises
    assert first.output_ndim == second.input_ndim == 3
    with pytest.raises(ValueError, match="coordinate system mismatch"):
        first.then(second, world, other)


def test_then_rejects_a_wrong_intermediate_system():
    _, world, first = uc1()
    second_world = WorldCoordinateSystem(
        name="world2", axes=(space("Z2"), space("Y2"), space("X2"))
    )
    second = AffineTransform.from_axis_map(
        world, second_world, axis_map={"Z": "Z2", "Y": "Y2", "X": "X2"}
    )
    with pytest.raises(ValueError, match="intermediate_coordinate_system must be"):
        first.then(second, second_world, second_world)


def test_then_propagates_broadcast_axes_through_the_second_matrix():
    """A | identity must not quietly lose the unbounded extent."""
    data, world, first = uc3()
    world2 = WorldCoordinateSystem(
        name="world2",
        axes=(
            Axis(name="T2", axis_type="time", unit="second"),
            Axis(name="C2", axis_type="channel"),
            space("Z2"),
            space("Y2"),
            space("X2"),
        ),
    )
    second = AffineTransform.from_axis_map(
        world,
        world2,
        axis_map={"T": "T2", "C": "C2", "Z": "Z2", "Y": "Y2", "X": "X2"},
    )
    chained = first.then(second, world, world2)
    assert chained.broadcast_axes == frozenset({world2.axis_by_name("C2").id})

    box = AxisAlignedBoundingBox(
        coordinate_system=data.id,
        min_coordinate=np.zeros(4),
        max_coordinate=np.ones(4),
    )
    mapped = chained.map_bounding_box(box, world2)
    assert mapped.min_coordinate[1] == -np.inf
    assert mapped.max_coordinate[1] == np.inf


def test_then_propagates_broadcast_through_an_axis_that_mixes_it_in():
    """Anything downstream of an unbounded axis is itself unbounded."""
    _, world, first = uc3()
    world2 = WorldCoordinateSystem(
        name="world2",
        axes=(
            Axis(name="T2", axis_type="time", unit="second"),
            Axis(name="C2", axis_type="channel"),
            space("Z2"),
            space("Y2"),
            space("X2"),
        ),
    )
    matrix = np.eye(6)
    matrix[2, 1] = 0.5  # Z2 = 0.5 * C + Z
    second = AffineTransform.from_matrix(
        matrix, world, world2, broadcast_output_axes=["C2"]
    )
    chained = first.then(second, world, world2)
    assert chained.broadcast_axes == frozenset(
        {world2.axis_by_name("C2").id, world2.axis_by_name("Z2").id}
    )


def test_there_is_no_matmul_and_no_compose_and_no_or_d10():
    _, world, first = uc1()
    second_world = WorldCoordinateSystem(
        name="world2", axes=(space("Z2"), space("Y2"), space("X2"))
    )
    second = AffineTransform.from_axis_map(
        world, second_world, axis_map={"Z": "Z2", "Y": "Y2", "X": "X2"}
    )
    assert not hasattr(AffineTransform, "compose")
    with pytest.raises(TypeError):
        first @ second
    with pytest.raises(TypeError):
        first | second


# --- constructor guardrails (D18, D22, D23, D26) -----------------------


def test_axis_map_is_required_d22():
    data, world, _ = uc1()
    with pytest.raises(TypeError):
        AffineTransform.from_axis_map(data, world)


def test_an_unmapped_input_axis_raises_d22():
    data, world, _ = uc1()
    with pytest.raises(ValueError, match="Every input axis"):
        AffineTransform.from_axis_map(data, world, axis_map={"z": "Z", "y": "Y"})


def test_an_output_axis_claimed_twice_raises():
    data, world, _ = uc1()
    with pytest.raises(ValueError, match="claimed by more than one"):
        AffineTransform.from_axis_map(
            data, world, axis_map={"z": "Z", "y": "Z", "x": "X"}
        )


def test_mapping_time_to_space_raises_d26():
    data = DataCoordinateSystem(
        name="d",
        datastore_id=uuid4(),
        axes=(Axis(name="t", axis_type="time"), space("z")),
    )
    world = WorldCoordinateSystem(axes=(space("Z"), space("T")))
    with pytest.raises(ValueError, match="axis_type"):
        AffineTransform.from_axis_map(data, world, axis_map={"t": "T", "z": "Z"})


def test_mismatched_units_do_not_raise_d16():
    """Units are recorded, not enforced."""
    data = DataCoordinateSystem(
        name="d",
        datastore_id=uuid4(),
        axes=(Axis(name="z", axis_type="space", unit="micrometer"),),
    )
    world = WorldCoordinateSystem(
        axes=(Axis(name="Z", axis_type="space", unit="millimeter"),)
    )
    assert AffineTransform.from_axis_map(data, world, axis_map={"z": "Z"})


def test_an_ambiguous_axis_name_raises_and_the_id_works_d6():
    first, second = space("z"), space("z")
    data = CoordinateSystem(name="d", axes=(first, second))
    world = CoordinateSystem(name="w", axes=(space("Z"), space("Y")))
    with pytest.raises(ValueError, match="ambiguous"):
        AffineTransform.from_axis_map(data, world, axis_map={"z": "Z", second.id: "Y"})
    assert AffineTransform.from_axis_map(
        data, world, axis_map={first.id: "Z", second.id: "Y"}
    )


def test_from_matrix_rejects_a_shape_that_disagrees_with_the_systems():
    data, world, _ = uc1()
    with pytest.raises(ValueError, match="must have shape"):
        AffineTransform.from_matrix(np.eye(5), data, world)


def test_a_non_finite_matrix_is_rejected_d44():
    data, world, _ = uc1()
    bad = np.eye(4)
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        AffineTransform.from_matrix(bad, data, world)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        AffineTransform.from_matrix(bad, data, world)


def test_there_is_no_identity_constructor_d18():
    assert not hasattr(AffineTransform, "identity")
    assert not hasattr(AffineTransform, "from_scale")


def test_the_axis_map_is_not_stored_d23():
    assert "axis_map" not in AffineTransform.model_fields


# --- the rendered -> world embedding (D34, D35, D36) -------------------


def tczyx_world():
    return WorldCoordinateSystem(
        axes=(
            Axis(name="T", axis_type="time", unit="second"),
            Axis(name="C", axis_type="channel"),
            space("Z"),
            space("Y"),
            space("X"),
        )
    )


def rendered_to_world(displayed, slices):
    world = tczyx_world()
    rendered = RenderedCoordinateSystem.from_world(world, displayed, uuid4())
    transform = AffineTransform.from_axis_map(
        rendered,
        world,
        axis_map={name: name for name in displayed},
        constant_output_axes=slices,
    )
    return world, rendered, transform


def test_the_3d_embedding_is_invertible_and_drops_the_sliced_axes_d34():
    _, _, transform = rendered_to_world(("Z", "Y", "X"), {"T": 7.0, "C": 2.0})
    assert transform.matrix.shape == (6, 4)
    assert np.linalg.matrix_rank(transform.linear) == 3
    inverse = transform.inverse()
    assert inverse is not None

    point = np.array([3.0, 4.0, 5.0])
    world_point = transform.map_coordinates(point)
    assert np.allclose(world_point, [7.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(transform.imap_coordinates(world_point), point)


def test_the_2d_embedding_is_an_exact_left_inverse_d34():
    _, _, transform = rendered_to_world(("Z", "Y"), {"T": 7.0, "C": 2.0, "X": 1.0})
    assert transform.matrix.shape == (6, 3)
    point = np.array([3.0, 4.0])
    mapped = transform.map_coordinates(point)
    assert np.allclose(transform.imap_coordinates(mapped), point)


def test_declaring_the_transform_world_to_rendered_gives_no_inverse_d34():
    """Pinned so the direction choice cannot be quietly reversed."""
    world, rendered, _ = rendered_to_world(("Z", "Y", "X"), {"T": 7.0, "C": 2.0})
    reversed_matrix = np.zeros((4, 6))
    reversed_matrix[0, 2] = 1.0
    reversed_matrix[1, 3] = 1.0
    reversed_matrix[2, 4] = 1.0
    reversed_matrix[3, 5] = 1.0
    backwards = AffineTransform.from_matrix(reversed_matrix, world, rendered)
    assert backwards.inverse() is None


def test_a_world_point_off_the_slice_maps_onto_it_d36():
    """A transform is a total function; a selection is a predicate."""
    _, _, transform = rendered_to_world(("Z", "Y", "X"), {"T": 7.0, "C": 2.0})
    on_slice = np.array([7.0, 2.0, 3.0, 4.0, 5.0])
    off_slice = np.array([99.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(
        transform.imap_coordinates(on_slice), transform.imap_coordinates(off_slice)
    )


def test_constant_output_axes_do_not_become_broadcast_axes_d35():
    _, _, transform = rendered_to_world(("Z", "Y", "X"), {"T": 7.0, "C": 2.0})
    assert transform.broadcast_axes == frozenset()
    assert transform.translation[0] == 7.0
    assert transform.translation[1] == 2.0


def test_broadcast_output_axes_do_both_d35():
    _, world, transform = uc3()
    assert transform.broadcast_axes == frozenset({world.axis_by_name("C").id})
    assert transform.translation[1] == 0.0


def test_an_output_axis_cannot_be_both_broadcast_and_constant():
    world = tczyx_world()
    rendered = RenderedCoordinateSystem.from_world(world, ("Z", "Y", "X"), uuid4())
    with pytest.raises(ValueError, match="both broadcast and constant"):
        AffineTransform.from_axis_map(
            rendered,
            world,
            axis_map={"Z": "Z", "Y": "Y", "X": "X"},
            broadcast_output_axes=["C"],
            constant_output_axes={"C": 2.0, "T": 7.0},
        )


def test_a_mapped_output_axis_cannot_also_be_declared():
    world = tczyx_world()
    rendered = RenderedCoordinateSystem.from_world(world, ("Z", "Y", "X"), uuid4())
    with pytest.raises(ValueError, match="both mapped and listed"):
        AffineTransform.from_axis_map(
            rendered,
            world,
            axis_map={"Z": "Z", "Y": "Y", "X": "X"},
            broadcast_output_axes=["Z"],
            constant_output_axes={"T": 7.0, "C": 2.0},
        )


# --- D46: permuting the rendered axes transposes the view --------------


def _node_matrix_for_rendered_order(displayed):
    """Compose visual -> data -> world -> rendered, as the renderer does.

    Design section 3.14's worked case: a ``zyx`` image in a ``ZYX``
    world with scales ``z=2.0, y=0.5, x=0.25``, displaying ``Y``/``X``
    sliced at ``Z``.  The only thing that varies between calls is the
    order of ``displayed``.
    """
    data = DataCoordinateSystem(
        name="image",
        datastore_id=uuid4(),
        axes=(space("z"), space("y"), space("x")),
    )
    world = WorldCoordinateSystem(axes=(space("Z"), space("Y"), space("X")))
    data_to_world = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={"z": "Z", "y": "Y", "x": "X"},
        scale={"z": 2.0, "y": 0.5, "x": 0.25},
    )

    rendered = RenderedCoordinateSystem.from_world(world, displayed, uuid4())
    rendered_to_world_transform = AffineTransform.from_axis_map(
        rendered,
        world,
        axis_map={name: name for name in displayed},
        constant_output_axes={"Z": 14.0},
    )

    visual = VisualCoordinateSystem.from_data(data, ("y", "x"), uuid4())
    visual_to_data = AffineTransform.from_axis_map(
        visual,
        data,
        axis_map={"y": "y", "x": "x"},
        constant_output_axes={"z": 7.0},
    )

    inverse = rendered_to_world_transform.inverse()
    assert inverse is not None, "the D34 embedding must have an exact left inverse"
    node = visual_to_data.then(data_to_world, data, world).then(
        inverse, world, rendered
    )
    return node


def test_a_permuted_rendered_system_transposes_the_node_matrix_d46():
    """The call sites differ only in the order of the displayed sequence."""
    in_order = _node_matrix_for_rendered_order(("Y", "X"))
    permuted = _node_matrix_for_rendered_order(("X", "Y"))

    assert np.allclose(in_order.linear, [[0.5, 0.0], [0.0, 0.25]])
    assert np.allclose(permuted.linear, [[0.0, 0.25], [0.5, 0.0]])


def test_a_permuted_rendered_system_has_a_negative_determinant_d46():
    """A swap of two axes is a reflection, not a rotation.

    Worth pinning because "rotate the view 90 degrees" is a *different*
    operation -- a swap plus a flip -- which an axis map cannot express
    without a negative scale.  Nothing in ``transform_v2`` cares; the
    renderer's winding order might.
    """
    in_order = _node_matrix_for_rendered_order(("Y", "X"))
    permuted = _node_matrix_for_rendered_order(("X", "Y"))

    assert np.linalg.det(in_order.linear) == pytest.approx(0.125)
    assert np.linalg.det(permuted.linear) == pytest.approx(-0.125)


def test_a_permuted_rendered_system_is_still_exactly_invertible_d46():
    permuted = _node_matrix_for_rendered_order(("X", "Y"))
    inverse = permuted.inverse()
    assert inverse is not None

    point = np.array([3.0, 4.0])
    assert np.allclose(inverse.map_coordinates(permuted.map_coordinates(point)), point)


# --- equality (D21) ----------------------------------------------------


def test_independently_built_identical_transforms_compare_equal_d21():
    data, world, first = uc1()
    second = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={"z": "Z", "y": "Y", "x": "X"},
        scale={"z": 0.5, "y": 0.1, "x": 0.1},
    )
    assert first.id != second.id
    assert first == second
    assert hash(first) == hash(second)


def test_equality_against_a_foreign_type_is_false_not_an_exception_d21():
    _, _, transform = uc1()
    assert (transform == "not a transform") is False
    assert (transform != "not a transform") is True


def test_transforms_differing_only_in_broadcast_axes_are_not_equal():
    data, world, with_broadcast = uc3()
    without = AffineTransform.from_matrix(
        with_broadcast.matrix,
        data,
        world,
        broadcast_output_axes=[],
    )
    assert np.array_equal(with_broadcast.matrix, without.matrix)
    assert with_broadcast != without


def test_hash_works_on_a_model_holding_an_unhashable_affine_d21():
    _, _, transform = uc1()
    with pytest.raises(TypeError):
        hash(transform.transform)
    assert isinstance(hash(transform), int)
