// Cell -> brick rule for the 3D multiscale brick LUT.
//
// Which level-k brick owns a base LUT cell.  This must give the same answer as
// cellier.render.lut_indirection._cell_brick_rule, which writes the LUT:
//
//     brick = min(cell / span_k, bricks_k - 1)
//
// span_k (base cells per level-k brick) and bricks_k (level-k brick count) are
// integers computed in Python and uploaded in u_block_scales, so the shader
// never rounds a scale itself.  Positions inside the brick still divide by the
// float scale (get_lod_scale).  Deriving the corner from the float scale
// instead -- floor(cell / scale), or floor(voxel / scale / block_size) --
// disagrees with the LUT on pyramids whose level ratio is not an integer and
// samples the wrong brick.  See docs/Explanations/multiscale_brick_lookup.md.
//
// Requires u_vol_params.block_size_{x,y,z} and u_block_scales.span_k/bricks_k.

fn get_brick_span(lut_w: u32) -> vec3<i32> {
    switch lut_w {
        case 1u:  { return vec3<i32>(u_block_scales.span_1.xyz); }
        case 2u:  { return vec3<i32>(u_block_scales.span_2.xyz); }
        case 3u:  { return vec3<i32>(u_block_scales.span_3.xyz); }
        case 4u:  { return vec3<i32>(u_block_scales.span_4.xyz); }
        case 5u:  { return vec3<i32>(u_block_scales.span_5.xyz); }
        case 6u:  { return vec3<i32>(u_block_scales.span_6.xyz); }
        case 7u:  { return vec3<i32>(u_block_scales.span_7.xyz); }
        case 8u:  { return vec3<i32>(u_block_scales.span_8.xyz); }
        case 9u:  { return vec3<i32>(u_block_scales.span_9.xyz); }
        default:  { return vec3<i32>(1); }
    }
}

fn get_brick_count(lut_w: u32) -> vec3<i32> {
    switch lut_w {
        case 1u:  { return vec3<i32>(u_block_scales.bricks_1.xyz); }
        case 2u:  { return vec3<i32>(u_block_scales.bricks_2.xyz); }
        case 3u:  { return vec3<i32>(u_block_scales.bricks_3.xyz); }
        case 4u:  { return vec3<i32>(u_block_scales.bricks_4.xyz); }
        case 5u:  { return vec3<i32>(u_block_scales.bricks_5.xyz); }
        case 6u:  { return vec3<i32>(u_block_scales.bricks_6.xyz); }
        case 7u:  { return vec3<i32>(u_block_scales.bricks_7.xyz); }
        case 8u:  { return vec3<i32>(u_block_scales.bricks_8.xyz); }
        case 9u:  { return vec3<i32>(u_block_scales.bricks_9.xyz); }
        default:  { return vec3<i32>(1); }
    }
}

// Level-k corner, in level-k voxels, of the brick that owns base cell `cell`.
fn brick_corner_from_cell(cell: vec3<i32>, lut_w: u32) -> vec3<f32> {
    let block_size = vec3<f32>(u_vol_params.block_size_x,
                               u_vol_params.block_size_y,
                               u_vol_params.block_size_z);
    let brick = min(cell / get_brick_span(lut_w),
                    get_brick_count(lut_w) - vec3<i32>(1));
    return vec3<f32>(max(brick, vec3<i32>(0))) * block_size;
}
