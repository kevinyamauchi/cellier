"""Cellier v2 GUI widgets for visual (appearance) controls."""

from cellier.gui.qt.visuals._colormap import QtColormapComboBox
from cellier.gui.qt.visuals._contrast_limits import QtClimRangeSlider
from cellier.gui.qt.visuals._image import QtVolumeRenderControls
from cellier.gui.qt.visuals._lod_bias import QtLodBiasSlider

__all__ = [
    "QtClimRangeSlider",
    "QtColormapComboBox",
    "QtLodBiasSlider",
    "QtVolumeRenderControls",
]
