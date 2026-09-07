"""Appearance controls specific to mesh visuals (anywidget).

Layer 3 of the three-layer design in ``plans/convenience_cleanup.md`` section
10.2: each class is a field name, a label, and the per-field defaults.  All
behaviour -- the bus contract, the echo filter, the fan-out over an
``OrthoViewer``'s panel group -- lives in the layer-1 base, and the control
itself in the layer-2 type.
"""

from __future__ import annotations

from typing import ClassVar

from cellier.gui.anywidget.visuals._base import (
    AnywidgetChoice,
    AnywidgetFloatSpin,
    AnywidgetToggle,
)


class AnywidgetSideCombo(AnywidgetChoice):
    """Which face windings are drawn.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.side``.
    """

    _field: ClassVar[str] = "side"
    _label: ClassVar[str] = "Side"
    _default_value: ClassVar[str] = "both"
    _default_choices: ClassVar[tuple[str, ...]] = ("both", "front", "back")


class AnywidgetWireframeToggle(AnywidgetToggle):
    """Draw the mesh as edges only.  Flat meshes.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.wireframe``.
    """

    _field: ClassVar[str] = "wireframe"
    _label: ClassVar[str] = "Wireframe"
    _default_value: ClassVar[bool] = False


class AnywidgetWireframeThicknessSpin(AnywidgetFloatSpin):
    """Edge thickness of the mesh wireframe, in screen pixels.

    Below 0.1 the wireframe is invisible; above roughly 20 it
    swamps the mesh.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.wireframe_thickness``.
    """

    _field: ClassVar[str] = "wireframe_thickness"
    _label: ClassVar[str] = "Wireframe thickness"
    _default_value: ClassVar[float] = 1.0
    _default_range: ClassVar[tuple[float, float]] = (0.1, 20.0)
    _default_step: ClassVar[float] = 0.1


class AnywidgetShininessSpin(AnywidgetFloatSpin):
    """Phong specular exponent.  Phong meshes.

    128 is the conventional practical ceiling.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.shininess``.
    """

    _field: ClassVar[str] = "shininess"
    _label: ClassVar[str] = "Shininess"
    _default_value: ClassVar[float] = 30.0
    _default_range: ClassVar[tuple[float, float]] = (0.0, 128.0)
    _default_step: ClassVar[float] = 1.0


class AnywidgetFlatShadingToggle(AnywidgetToggle):
    """Use face normals instead of smooth vertex normals.  Phong meshes.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.flat_shading``.
    """

    _field: ClassVar[str] = "flat_shading"
    _label: ClassVar[str] = "Flat shading"
    _default_value: ClassVar[bool] = False
