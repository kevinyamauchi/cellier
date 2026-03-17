import numpy as np

from block_volume.layout import BlockLayout
from block_volume.uniforms import LUT_PARAMS_DTYPE, build_lut_params_buffer


class TestLutParamsDtype:
    def test_field_count(self):
        assert len(LUT_PARAMS_DTYPE.names) == 16

    def test_total_size_is_64_bytes(self):
        # 4 vec4<f32> = 4 * 16 = 64 bytes
        assert LUT_PARAMS_DTYPE.itemsize == 64

    def test_all_fields_are_f32(self):
        for name in LUT_PARAMS_DTYPE.names:
            assert LUT_PARAMS_DTYPE[name] == np.dtype("<f4")


class TestBuildLutParamsBuffer:
    def test_buffer_data_shape_is_0d(self):
        """Buffer must be 0-d to avoid WGSL array wrapper (lesson L2)."""
        layout = BlockLayout.from_volume_shape((64, 64, 64), block_size=32)
        buf = build_lut_params_buffer(layout)
        assert buf.data.shape == ()

    def test_block_size_values(self):
        layout = BlockLayout.from_volume_shape((64, 64, 64), block_size=16)
        buf = build_lut_params_buffer(layout)
        data = buf.data
        assert data["block_size_x"] == 16.0
        assert data["block_size_y"] == 16.0
        assert data["block_size_z"] == 16.0

    def test_cache_size_uses_whd_order(self):
        """cache_size (x,y,z) should map to (pW, pH, pD)."""
        layout = BlockLayout.from_volume_shape((32, 64, 96), block_size=32)
        buf = build_lut_params_buffer(layout)
        data = buf.data
        pd, ph, pw = layout.padded_shape
        assert data["cache_size_x"] == float(pw)
        assert data["cache_size_y"] == float(ph)
        assert data["cache_size_z"] == float(pd)

    def test_lut_size_uses_whd_order(self):
        """lut_size (x,y,z) should map to (gW, gH, gD)."""
        layout = BlockLayout.from_volume_shape((32, 64, 96), block_size=32)
        buf = build_lut_params_buffer(layout)
        data = buf.data
        gd, gh, gw = layout.grid_dims
        assert data["lut_size_x"] == float(gw)
        assert data["lut_size_y"] == float(gh)
        assert data["lut_size_z"] == float(gd)

    def test_lut_offset_is_zero(self):
        layout = BlockLayout.from_volume_shape((64, 64, 64), block_size=32)
        buf = build_lut_params_buffer(layout)
        data = buf.data
        assert data["lut_offset_x"] == 0.0
        assert data["lut_offset_y"] == 0.0
        assert data["lut_offset_z"] == 0.0

    def test_padding_fields_are_zero(self):
        layout = BlockLayout.from_volume_shape((64, 64, 64), block_size=32)
        buf = build_lut_params_buffer(layout)
        data = buf.data
        for pad_name in ("_pad0", "_pad1", "_pad2", "_pad3"):
            assert data[pad_name] == 0.0
