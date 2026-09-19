// Cell -> tile rule for the 2D multiscale tile LUT.
//
// The 2D counterpart of cellier.brick_rule.wgsl: which level-k tile owns a base
// LUT cell.  Must give the same answer as
// cellier.render.lut_indirection._cell_brick_rule, which writes the LUT:
//
//     tile = min(cell / span_k, tiles_k - 1)
//
// span_k and tiles_k are integers uploaded in u_block_scales (fields span_k and
// bricks_k, shader order x=W, y=H).  Positions inside the tile still use the
// float scale (get_tile_scale).  Taking the tile corner from the float position
// instead -- (pos * scale) mod block_size -- disagrees with the LUT on pyramids
// whose level ratio is not an integer.  See
// docs/Explanations/multiscale_brick_lookup.md.
//
// Requires u_lut_params.block_size_{x,y} and u_block_scales.span_k/bricks_k.

fn get_tile_span(level: i32) -> vec2<i32> {
    switch level {
        case 1: { return vec2<i32>(u_block_scales.span_1.xy); }
        case 2: { return vec2<i32>(u_block_scales.span_2.xy); }
        case 3: { return vec2<i32>(u_block_scales.span_3.xy); }
        case 4: { return vec2<i32>(u_block_scales.span_4.xy); }
        case 5: { return vec2<i32>(u_block_scales.span_5.xy); }
        case 6: { return vec2<i32>(u_block_scales.span_6.xy); }
        case 7: { return vec2<i32>(u_block_scales.span_7.xy); }
        case 8: { return vec2<i32>(u_block_scales.span_8.xy); }
        case 9: { return vec2<i32>(u_block_scales.span_9.xy); }
        default: { return vec2<i32>(1, 1); }
    }
}

fn get_tile_count(level: i32) -> vec2<i32> {
    switch level {
        case 1: { return vec2<i32>(u_block_scales.bricks_1.xy); }
        case 2: { return vec2<i32>(u_block_scales.bricks_2.xy); }
        case 3: { return vec2<i32>(u_block_scales.bricks_3.xy); }
        case 4: { return vec2<i32>(u_block_scales.bricks_4.xy); }
        case 5: { return vec2<i32>(u_block_scales.bricks_5.xy); }
        case 6: { return vec2<i32>(u_block_scales.bricks_6.xy); }
        case 7: { return vec2<i32>(u_block_scales.bricks_7.xy); }
        case 8: { return vec2<i32>(u_block_scales.bricks_8.xy); }
        case 9: { return vec2<i32>(u_block_scales.bricks_9.xy); }
        default: { return vec2<i32>(1, 1); }
    }
}

// Level-k corner, in level-k pixels, of the tile that owns base cell `cell`.
fn tile_corner_from_cell(cell: vec2<i32>, level: i32) -> vec2<f32> {
    let block_size = vec2<f32>(u_lut_params.block_size_x, u_lut_params.block_size_y);
    let tile = min(cell / get_tile_span(level), get_tile_count(level) - vec2<i32>(1));
    return vec2<f32>(max(tile, vec2<i32>(0, 0))) * block_size;
}
