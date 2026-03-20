{$ include 'pygfx.std.wgsl' $}
$$ if colormap_dim
    {$ include 'pygfx.colormap.wgsl' $}
$$ endif
{$ include 'pygfx.light_phong_simple.wgsl' $}
{$ include 'pygfx.volume_common.wgsl' $}

// Binding declarations are injected by define_bindings -> bindings_code.
//
// t_img   : proxy texture (volume-sized) — used by get_vol_geometry() for the
//           bounding box and by fs_main for sizef = textureDimensions(t_img).
// t_cache : actual fixed-size brick cache texture (for sampling).
// s_cache : sampler for t_cache.
// t_lut   : texture_3d<u32> — RGBA8UI brick address table.
// u_lut_params   : LutParams uniform (block_size, cache_size, lut_size, lut_offset).
// u_block_scales : BlockScales uniform (10 × vec4 per-level scale factors).


// ── Block-scale lookup ────────────────────────────────────────────────

fn get_block_scale(level: u32) -> vec4<f32> {
    switch level {
        case 1u:  { return u_block_scales.scale_1; }
        case 2u:  { return u_block_scales.scale_2; }
        case 3u:  { return u_block_scales.scale_3; }
        case 4u:  { return u_block_scales.scale_4; }
        case 5u:  { return u_block_scales.scale_5; }
        case 6u:  { return u_block_scales.scale_6; }
        case 7u:  { return u_block_scales.scale_7; }
        case 8u:  { return u_block_scales.scale_8; }
        case 9u:  { return u_block_scales.scale_9; }
        default:  { return u_block_scales.scale_0; }
    }
}


// ── Vertex stage ──────────────────────────────────────────────────────

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
};

@vertex
fn vs_main(in: VertexInput) -> Varyings {
    var geo = get_vol_geometry();
    let index = i32(in.vertex_index);
    let i0 = geo.indices[index];

    let data_pos  = vec4<f32>(geo.positions[i0], 1.0);
    let world_pos = u_wobject.world_transform * data_pos;
    let ndc_pos   = u_stdinfo.projection_transform
                  * u_stdinfo.cam_transform
                  * world_pos;

    let ndc_to_data = u_wobject.world_transform_inv
                    * u_stdinfo.cam_transform_inv
                    * u_stdinfo.projection_transform_inv;

    let cam_sign = sign(
        u_stdinfo.cam_transform[0][0] *
        u_stdinfo.cam_transform[1][1] *
        u_stdinfo.cam_transform[2][2]
    );

    var varyings: Varyings;
    varyings.position      = vec4<f32>(ndc_pos);
    varyings.world_pos     = vec3<f32>(world_pos.xyz);
    varyings.data_back_pos = vec4<f32>(data_pos);
    varyings.data_near_pos = vec4<f32>(
        ndc_to_data * vec4<f32>(ndc_pos.xy, -1.0 * cam_sign * ndc_pos.w, ndc_pos.w)
    );
    varyings.data_far_pos = vec4<f32>(
        ndc_to_data * vec4<f32>(ndc_pos.xy, cam_sign * ndc_pos.w, ndc_pos.w)
    );
    return varyings;
}


// ── LUT-based sampling ────────────────────────────────────────────────

fn sample_vol_lut(texcoord: vec3<f32>, sizef: vec3<f32>) -> vec4<f32> {
    let block_size = vec3<f32>(
        u_lut_params.block_size_x,
        u_lut_params.block_size_y,
        u_lut_params.block_size_z,
    );
    let cache_size = vec3<f32>(
        u_lut_params.cache_size_x,
        u_lut_params.cache_size_y,
        u_lut_params.cache_size_z,
    );
    let lut_size = vec3<f32>(
        u_lut_params.lut_size_x,
        u_lut_params.lut_size_y,
        u_lut_params.lut_size_z,
    );
    let overlap = u_lut_params.overlap;
    let padded_size = block_size + vec3<f32>(2.0 * overlap);

    let pos = clamp(texcoord * sizef, vec3<f32>(0.0), sizef - vec3<f32>(0.5));

    let brick_f   = floor(pos / block_size);
    let brick_idx = vec3<i32>(brick_f);

    let lut_max  = vec3<i32>(lut_size) - vec3<i32>(1, 1, 1);
    let safe_idx = clamp(brick_idx, vec3<i32>(0, 0, 0), lut_max);

    let lutv = textureLoad(t_lut, safe_idx, 0);

    let tile_origin   = vec3<f32>(lutv.xyz) * padded_size;
    let sj            = get_block_scale(lutv.w).xyz;
    let scaled_pos    = pos * sj;
    let within_coarse = scaled_pos - floor(scaled_pos / block_size) * block_size;
    let cache_pos     = tile_origin + within_coarse + vec3<f32>(overlap);
    let cache_coord   = cache_pos / cache_size;

    return textureSample(t_cache, s_cache, cache_coord);
}


// ── Fragment stage — isosurface raycast ───────────────────────────────

@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    {$ include 'pygfx.clipping_planes.wgsl' $}

    // sizef: proxy-texture data-space dimensions.
    // The proxy may use grid dims (1 texel/brick) to avoid a huge CPU alloc.
    // ALL ray-geometry math (clipping, start_coord, step_coord, depth) uses
    // sizef so it is self-consistent with the data coords from vs_main.
    let sizef = vec3<f32>(textureDimensions(t_img));

    // vol_sizef: the real padded voxel extent of the volume.
    // Used only to calibrate step density and to compute the ratio between
    // the proxy coordinate unit and a voxel.
    let vol_sizef = vec3<f32>(
        u_lut_params.vol_size_x,
        u_lut_params.vol_size_y,
        u_lut_params.vol_size_z,
    );

    // How many voxels fit in one proxy texel (= block_size when proxy uses
    // grid dims; = 1.0 when proxy has full voxel resolution).
    let voxels_per_unit = vol_sizef.x / sizef.x;

    // Calibrate step density against the real voxel count so the number of
    // ray steps is the same regardless of proxy resolution.
    let relative_step_size = clamp(
        sqrt(max(vol_sizef.x, max(vol_sizef.y, vol_sizef.z))) / 20.0,
        0.1,
        0.8,
    );

    let back_pos  = varyings.data_back_pos.xyz / varyings.data_back_pos.w;
    let far_pos   = varyings.data_far_pos.xyz  / varyings.data_far_pos.w;
    let near_pos  = varyings.data_near_pos.xyz / varyings.data_near_pos.w;
    let view_ray  = normalize(far_pos - near_pos);

    var dist = dot(near_pos - back_pos, view_ray);
    dist = max(dist, min((-0.5 - back_pos.x) / view_ray.x, (sizef.x - 0.5 - back_pos.x) / view_ray.x));
    dist = max(dist, min((-0.5 - back_pos.y) / view_ray.y, (sizef.y - 0.5 - back_pos.y) / view_ray.y));
    dist = max(dist, min((-0.5 - back_pos.z) / view_ray.z, (sizef.z - 0.5 - back_pos.z) / view_ray.z));

    let front_pos = back_pos + view_ray * dist;
    // Scale dist (in proxy/grid-unit space) to voxel-equivalent distance
    // before computing nsteps, so step density matches voxel-space behaviour.
    let nsteps    = i32(-dist * voxels_per_unit / relative_step_size + 0.5);
    if (nsteps < 1) { discard; }
    let nstepsf = f32(nsteps);

    let start_coord = (front_pos + vec3<f32>(0.5, 0.5, 0.5)) / sizef;
    let step_coord  = ((back_pos - front_pos) / sizef) / f32(nsteps);

    let iso_threshold = u_material.threshold;
    let actual_step_coord = u_material.step_size * step_coord;
    var surface_found = false;
    var the_coord = start_coord;
    var the_value: vec4<f32>;

    for (var iter = 0.0; iter < nstepsf; iter = iter + 1.0) {
        let coord = start_coord + iter * actual_step_coord;
        let value = sample_vol_lut(coord, sizef);
        if (value.r > iso_threshold) {
            the_coord = coord;
            the_value = value;
            surface_found = true;
            break;
        }
    }

    if (surface_found) {
        let substep_coord = -1.0 * u_material.substep_size * step_coord;
        let substep_start = the_coord;
        let max_iter = 1.0 / u_material.substep_size;
        for (var iter2 = 1.0; iter2 < max_iter; iter2 = iter2 + 1.0) {
            let coord = substep_start + iter2 * substep_coord;
            let value = sample_vol_lut(coord, sizef);
            if (value.r < iso_threshold) {
                break;
            }
            the_coord = coord;
            the_value = value;
        }
    } else {
        discard;
    }

    let color = sampled_value_to_color(the_value);
    $$ if colorspace == 'srgb'
        let physical_color = srgb2physical(color.rgb);
    $$ else
        let physical_color = color.rgb;
    $$ endif

    let gradient_coord = 1.5 * step_coord;
    var normal: vec3<f32>;

    let neg_x = sample_vol_lut(the_coord + vec3<f32>(-gradient_coord.x, 0.0, 0.0), sizef);
    let pos_x = sample_vol_lut(the_coord + vec3<f32>( gradient_coord.x, 0.0, 0.0), sizef);
    normal.x = pos_x.r - neg_x.r;

    let neg_y = sample_vol_lut(the_coord + vec3<f32>(0.0, -gradient_coord.y, 0.0), sizef);
    let pos_y = sample_vol_lut(the_coord + vec3<f32>(0.0,  gradient_coord.y, 0.0), sizef);
    normal.y = pos_y.r - neg_y.r;

    let neg_z = sample_vol_lut(the_coord + vec3<f32>(0.0, 0.0, -gradient_coord.z), sizef);
    let pos_z = sample_vol_lut(the_coord + vec3<f32>(0.0, 0.0,  gradient_coord.z), sizef);
    normal.z = pos_z.r - neg_z.r;

    normal = normalize(normal);

    let view_direction = normalize(step_coord);
    let is_front = dot(normal, view_direction) > 0.0;
    let lighted_color = lighting_phong(is_front, normal, view_direction, physical_color);

    let opacity = color.a * u_material.opacity;

    do_alpha_test(opacity);

    let data_pos  = the_coord * sizef - vec3<f32>(0.5, 0.5, 0.5);
    let world_pos = u_wobject.world_transform * vec4<f32>(data_pos, 1.0);
    let ndc_pos   = u_stdinfo.projection_transform
                  * u_stdinfo.cam_transform
                  * world_pos;

    var out: FragmentOutput;
    out.color = vec4<f32>(lighted_color, opacity);
    out.depth = ndc_pos.z / ndc_pos.w;
    return out;
}
