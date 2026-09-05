"""Qt panel for the ambient occlusion settings.

The setting is named ``ambient_occlusion`` and the pass that implements it
is named ``SSAOPass``.  That is deliberate: SSAO is the algorithm, and the
config is named for what a user is asking for rather than for how it is
computed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from cellier.gui._render_controls import (
    RADIUS_SLIDER_HEADROOM,
    RENDER_SECTION_TITLES,
)
from cellier.gui.qt.render._base import QtRenderConfigPanel

if TYPE_CHECKING:
    from collections.abc import Callable

    from cellier.render._config import AmbientOcclusionConfig


class QtAmbientOcclusionControls(QtRenderConfigPanel):
    """Ambient occlusion settings on the cellier bus.

    Ambient occlusion darkens creases by sampling the depth buffer, and is
    the cheapest shape cue available for cellier's default unlit
    isosurfaces.  Two states this panel reports but does not control:

    * **The pass is off in 2D**, whatever ``enabled`` says -- a 2D cellier
      scene is a plane at near-constant depth, where the occlusion comes
      out uniform.
    * **The radius may be auto-derived.**  A fixed default is meaningless
      across cellier's coordinate systems, so the default is a fraction of
      the scene bounding box diagonal.  The effective-radius readout is
      what makes the number comparable to the thing being rendered, which
      is the whole difficulty with this control.

    Parameters
    ----------
    config :
        The live ``AmbientOcclusionConfig`` to read initial values from -- typically
        ``controller.render_config.ambient_occlusion``.
    effective_radius :
        Called to read the radius actually in use, in scene units, for the
        readout.  Typically ``lambda: controller.ambient_occlusion_effective_radius``.
        ``None`` omits the readout.
    parent :
        Optional Qt parent widget.
    """

    section: ClassVar[str] = "ambient_occlusion"
    title: ClassVar[str] = RENDER_SECTION_TITLES["ambient_occlusion"]

    def __init__(
        self,
        config: AmbientOcclusionConfig,
        *,
        effective_radius: Callable[[], float | None] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._effective_radius = effective_radius
        self._config = config

        # The radius pair needs behaviour the spec cannot express -- a mode
        # toggle that writes the same field, and a slider whose range comes
        # from the scene -- so the panel builds that group itself.
        self._build_from_spec(config, skip={"radius", "auto_radius_fraction"})
        self._build_radius_group(config)

    # ------------------------------------------------------------------
    # The radius group
    # ------------------------------------------------------------------

    def _build_radius_group(self, config: AmbientOcclusionConfig) -> None:
        """The radius, its auto/explicit mode, and what it came out as."""
        from cellier.gui._render_controls import RENDER_CONTROLS

        specs = {c.field: c for c in RENDER_CONTROLS[self.section]}

        def _build(_parent) -> None:
            from qtpy.QtWidgets import QCheckBox

            effective = self._read_effective_radius(config)

            auto = QCheckBox("Derive from scene size", self._container)
            auto.setChecked(config.radius is None)
            auto.setToolTip(
                "A radius is in scene units, and a fixed default means "
                "nothing across cellier's coordinate systems -- a bounding "
                "box may be 96 units or 0.0003."
            )
            self._layout.addWidget(auto)

            fraction = self._add_spec_control(specs["auto_radius_fraction"], config)

            radius_spec = specs["radius"]
            radius = self._add_float_slider(
                "radius",
                radius_spec.label,
                config.radius if config.radius is not None else effective,
                radius_spec.minimum,
                max(effective * RADIUS_SLIDER_HEADROOM, 1.0),
                decimals=radius_spec.decimals,
                tooltip=self._tooltip(radius_spec),
            )

            def _apply_mode(is_auto: bool) -> None:
                fraction.setEnabled(is_auto)
                radius.setEnabled(not is_auto)

            def _on_auto(is_auto: bool) -> None:
                _apply_mode(is_auto)
                # ``radius = None`` *is* the auto mode; there is no separate
                # flag, so the checkbox writes the field either way.
                self._emit("radius", None if is_auto else radius.value())

            auto.toggled.connect(_on_auto)
            _apply_mode(auto.isChecked())

            # An inbound radius change also decides the mode, so the
            # checkbox has to follow it rather than owning it.
            radius_applier = self._appliers["radius"]

            def _apply_radius(value) -> None:
                is_auto = value is None
                auto.blockSignals(True)
                auto.setChecked(is_auto)
                auto.blockSignals(False)
                _apply_mode(is_auto)
                if value is not None:
                    radius_applier(value)

            self._register("radius", _apply_radius)

            if self._effective_radius is not None:
                self._add_readout(
                    "In use", lambda: _format_radius(self._effective_radius())
                )

        self._add_group("Radius", _build)

    def _read_effective_radius(self, config: AmbientOcclusionConfig) -> float:
        """The radius in use, falling back to the config when there is no canvas."""
        if self._effective_radius is not None:
            value = self._effective_radius()
            if value:
                return float(value)
        return float(config.radius) if config.radius else 1.0


def _format_radius(value: float | None) -> str:
    """Render the effective radius, or say why there is not one yet."""
    if not value:
        return "no canvas yet"
    return f"{value:.4g} scene units"
