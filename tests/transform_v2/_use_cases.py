"""The four use cases of design section 3, as raw (linear, translation) pairs.

Shared by test_geometry_ops.py, which works on arrays rather than models.
The matrices are lifted verbatim from
``scripts/transform_v2_geometry_ops_probe.py``.
"""

import numpy as np

# UC1  zyx -> ZYX, scale only
UC1 = (np.diag([0.5, 0.1, 0.1]), np.zeros(3), ())

# UC2  czyx -> cZYX, scale + translation on ZYX only
UC2 = (
    np.diag([1.0, 0.5, 0.1, 0.1]),
    np.array([0.0, 10.0, -2.0, 3.0]),
    (),
)

# UC3  tzyx -> TCZYX, broadcast over C (output index 1)
UC3 = (
    np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0.5, 0, 0],
            [0, 0, 0.1, 0],
            [0, 0, 0, 0.1],
        ],
        float,
    ),
    np.array([0.0, 0.0, 10.0, -2.0, 3.0]),
    (1,),
)

# UC4  zyx -> XYZ  (rows X, Y, Z; columns z, y, x)
UC4 = (
    np.array([[0, 0, 0.1], [0, 0.1, 0], [0.5, 0, 0]], float),
    np.zeros(3),
    (),
)

USE_CASES = {"UC1": UC1, "UC2": UC2, "UC3": UC3, "UC4": UC4}


# ---------------------------------------------------------------------
# The same four cases as models (design section 9.3), built through
# from_axis_map.  Each builder returns (data_system, world_system,
# transform) so a test can assert against the systems it was built from.
# ---------------------------------------------------------------------


def _space(name):
    from cellier.transform_v2 import Axis

    return Axis(name=name, axis_type="space", unit="micrometer")


def uc1():
    """zyx -> ZYX, scale only."""
    from uuid import uuid4

    from cellier.transform_v2 import (
        AffineTransform,
        DataCoordinateSystem,
        WorldCoordinateSystem,
    )

    data = DataCoordinateSystem(
        name="cells",
        datastore_id=uuid4(),
        axes=(_space("z"), _space("y"), _space("x")),
    )
    world = WorldCoordinateSystem(
        axes=(_space("Z"), _space("Y"), _space("X")),
    )
    transform = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={"z": "Z", "y": "Y", "x": "X"},
        scale={"z": 0.5, "y": 0.1, "x": 0.1},
    )
    return data, world, transform


def uc2():
    """czyx -> cZYX, scale + translation on ZYX only."""
    from uuid import uuid4

    from cellier.transform_v2 import (
        AffineTransform,
        Axis,
        DataCoordinateSystem,
        WorldCoordinateSystem,
    )

    data = DataCoordinateSystem(
        name="cells",
        datastore_id=uuid4(),
        axes=(
            Axis(name="c", axis_type="channel"),
            _space("z"),
            _space("y"),
            _space("x"),
        ),
    )
    world = WorldCoordinateSystem(
        axes=(
            Axis(name="c", axis_type="channel"),
            _space("Z"),
            _space("Y"),
            _space("X"),
        )
    )
    transform = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={"c": "c", "z": "Z", "y": "Y", "x": "X"},
        scale={"z": 0.5, "y": 0.1, "x": 0.1},
        translation={"z": 10.0, "y": -2.0, "x": 3.0},
    )
    return data, world, transform


def uc3():
    """tzyx -> TCZYX, broadcast over C."""
    from uuid import uuid4

    from cellier.transform_v2 import (
        AffineTransform,
        Axis,
        DataCoordinateSystem,
        WorldCoordinateSystem,
    )

    data = DataCoordinateSystem(
        name="timelapse",
        datastore_id=uuid4(),
        axes=(
            Axis(name="t", axis_type="time", unit="second"),
            _space("z"),
            _space("y"),
            _space("x"),
        ),
    )
    world = WorldCoordinateSystem(
        axes=(
            Axis(name="T", axis_type="time", unit="second"),
            Axis(name="C", axis_type="channel"),
            _space("Z"),
            _space("Y"),
            _space("X"),
        )
    )
    transform = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={"t": "T", "z": "Z", "y": "Y", "x": "X"},
        scale={"z": 0.5, "y": 0.1, "x": 0.1},
        translation={"z": 10.0, "y": -2.0, "x": 3.0},
        broadcast_output_axes=["C"],
    )
    return data, world, transform


def uc4():
    """zyx -> XYZ; the identical call to UC1, with a reordered world."""
    from uuid import uuid4

    from cellier.transform_v2 import (
        AffineTransform,
        DataCoordinateSystem,
        WorldCoordinateSystem,
    )

    data = DataCoordinateSystem(
        name="cells",
        datastore_id=uuid4(),
        axes=(_space("z"), _space("y"), _space("x")),
    )
    world = WorldCoordinateSystem(
        axes=(_space("X"), _space("Y"), _space("Z")),
    )
    transform = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={"z": "Z", "y": "Y", "x": "X"},
        scale={"z": 0.5, "y": 0.1, "x": 0.1},
    )
    return data, world, transform


MODEL_USE_CASES = {"UC1": uc1, "UC2": uc2, "UC3": uc3, "UC4": uc4}
