"""Appearance controls specific to label visuals (Qt).

Layer 3 of the three-layer design in ``plans/convenience_cleanup.md`` section
10.2: each class is a field name, a label, and the per-field defaults.  All
behaviour -- the bus contract, the echo filter, the fan-out over an
``OrthoViewer``'s panel group -- lives in the layer-1 base, and the control
itself in the layer-2 type.
"""

from __future__ import annotations

from typing import ClassVar

from cellier.gui.qt.visuals._base import (
    QtChoice,
    QtIntSpin,
)


class QtLabelsRenderModeCombo(QtChoice):
    """How label voxels are rendered in 3D.

    Distinct from the image render-mode control: the labels models
    spell this field ``iso_categorical``/``flat_categorical`` where
    the image models spell it ``mip``/``iso``/``minip``.  Pass
    ``choices=literal_choices(appearance, 'render_mode')`` so the
    in-memory and multiscale variants each offer their own set and
    ``gradient_debug`` is excluded (design section 6.5.1
    proposal 4).

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.render_mode``.
    """

    _field: ClassVar[str] = "render_mode"
    _label: ClassVar[str] = "Render mode"
    _default_value: ClassVar[str] = "iso_categorical"
    _default_choices: ClassVar[tuple[str, ...]] = (
        "iso_categorical",
        "flat_categorical",
    )


class QtSaltSpin(QtIntSpin):
    """Hash seed for the random label colormap.

    The number itself is meaningless -- only the colouring it
    produces matters -- so this control carries a shuffle button
    that writes a random seed in range.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.salt``.
    """

    _field: ClassVar[str] = "salt"
    _label: ClassVar[str] = "Salt"
    _default_value: ClassVar[int] = 0
    _default_range: ClassVar[tuple[int, int]] = (0, 2**31 - 1)
    _shuffle: ClassVar[bool] = True


class QtBackgroundLabelSpin(QtIntSpin):
    """Label id rendered as transparent.

    The range spans the label dtype: stores are ``int32`` and
    nothing forbids a negative id.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.background_label``.
    """

    _field: ClassVar[str] = "background_label"
    _label: ClassVar[str] = "Background label"
    _default_value: ClassVar[int] = 0
    _default_range: ClassVar[tuple[int, int]] = (-(2**31), 2**31 - 1)
