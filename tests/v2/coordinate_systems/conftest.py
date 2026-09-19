"""Fixtures for the coordinate-system tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.v2.data.test_ome_zarr_image_store import _write_synthetic_ome_zarr

if TYPE_CHECKING:
    import pathlib


@pytest.fixture
def ome_zarr_5d(tmp_path: pathlib.Path) -> str:
    """A synthetic 5-D ``tczyx`` OME-Zarr v0.5 store, as a ``file://`` URI."""
    store_path = tmp_path / "test.ome.zarr"
    _write_synthetic_ome_zarr(store_path)
    return f"file://{store_path}"
