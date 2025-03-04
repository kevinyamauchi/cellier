"""A workshop for rendering cellular models."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cellier")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Kevin Yamauchi"
__email__ = "kevin.yamauchi@gmail.com"
