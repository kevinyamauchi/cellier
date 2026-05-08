"""Tests for label colormap GPU resource builders."""

from __future__ import annotations

import warnings

import numpy as np


def test_build_direct_lut_two_entries():
    from cellier.v2.render.shaders._label_colormap import build_direct_lut_textures

    cd = {1: (1.0, 0.0, 0.0, 1.0), -3: (0.0, 0.0, 1.0, 1.0)}
    keys_tex, colors_tex, n = build_direct_lut_textures(cd)
    assert n == 2
    # Texture width should equal n_entries
    assert keys_tex.size[0] == 2
    # Keys must be sorted: [-3, 1]
    key_data = keys_tex.data.reshape(-1)
    assert key_data[0] == -3
    assert key_data[1] == 1
    # Colors must match sorted order
    color_data = colors_tex.data.reshape(2, 4)
    np.testing.assert_allclose(color_data[0], [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(color_data[1], [1.0, 0.0, 0.0, 1.0])


def test_build_direct_lut_empty():
    from cellier.v2.render.shaders._label_colormap import build_direct_lut_textures

    keys_tex, colors_tex, n = build_direct_lut_textures({})
    assert n == 0
    # Should return dummy 1-entry textures without error
    assert keys_tex is not None
    assert colors_tex is not None


def test_build_direct_lut_warns_over_limit():
    from cellier.v2.render.shaders._label_colormap import (
        _MAX_DIRECT_ENTRIES,
        build_direct_lut_textures,
    )

    oversized = {i: (0.0, 0.0, 0.0, 1.0) for i in range(_MAX_DIRECT_ENTRIES + 1)}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        keys_tex, colors_tex, n = build_direct_lut_textures(oversized)
    assert any("65536" in str(warning.message) for warning in w)
    assert n == _MAX_DIRECT_ENTRIES


def test_build_label_params_buffer_field_values():
    from cellier.v2.render.shaders._label_colormap import (
        LABEL_PARAMS_DTYPE,
        build_label_params_buffer,
    )

    buf = build_label_params_buffer(background_label=-5, salt=42, n_entries=7)
    data = np.frombuffer(buf.data, dtype=LABEL_PARAMS_DTYPE)[0]
    assert int(data["background_label"]) == -5
    assert int(data["salt"]) == 42
    assert int(data["n_entries"]) == 7


def test_build_label_params_buffer_dtype():
    from cellier.v2.render.shaders._label_colormap import (
        LABEL_PARAMS_DTYPE,
        build_label_params_buffer,
    )

    buf = build_label_params_buffer(background_label=0, salt=0, n_entries=0)
    assert buf.data.nbytes == LABEL_PARAMS_DTYPE.itemsize
    assert LABEL_PARAMS_DTYPE.itemsize == 16
