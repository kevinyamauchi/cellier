"""A workshop for rendering cellular models."""

from importlib.metadata import PackageNotFoundError, version

from cellier.controller import CellierController

__all__ = ["CellierController"]

try:
    __version__ = version("cellier")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Kevin Yamauchi"
__email__ = "kevin.yamauchi@gmail.com"
