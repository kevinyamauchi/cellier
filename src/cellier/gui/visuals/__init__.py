"""Cellier v2 GUI widgets for visual (appearance) controls."""

from cellier.gui.visuals._colormap import QtColormapComboBox
from cellier.gui.visuals._contrast_limits import QtClimRangeSlider
from cellier.gui.visuals._image import QtVolumeRenderControls

__all__ = ["QtColormapComboBox", "QtClimRangeSlider", "QtVolumeRenderControls"]
