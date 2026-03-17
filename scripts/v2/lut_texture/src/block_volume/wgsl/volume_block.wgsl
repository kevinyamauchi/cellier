{$ include 'pygfx.std.wgsl' $}
$$ if colormap_dim
    {$ include 'pygfx.colormap.wgsl' $}
$$ endif
{$ include 'pygfx.volume_common.wgsl' $}

// All @group/@binding declarations (t_img, s_img, t_lut, u_lut_params, etc.)
// are injected here by define_bindings -> bindings_code.
// t_lut is declared as texture_3d<u32>  (RGBA8UI source).
// u_lut_params is declared as the LutParams struct from LUT_PARAMS_DTYPE.


// Vertex stage
// Identical to the standard volume_ray.wgsl vertex shader.
// get_vol_geometry() reads textureDimensions(t_img) = cache dimensions,
// which is correct because cache size == padded volume size in phase 0.

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


// LUT-based sampling
// Replaces sample_vol() from pygfx.volume_common.wgsl.
//
// texcoord : normalised [0,1]^3 within the cache texture.
// sizef    : cache dimensions in voxels (from textureDimensions(t_img)).
//
// Coordinate convention throughout: (x=W, y=H, z=D) matching WGSL texture axes.

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

    // Voxel position in the cache.
    let pos = texcoord * sizef;

    // Which brick does this voxel fall into?
    let brick_f   = floor(pos / block_size);
    let brick_idx = vec3<i32>(brick_f);

    // Clamp to valid LUT range (boundary voxels of the padded region).
    let lut_max  = vec3<i32>(lut_size) - vec3<i32>(1, 1, 1);
    let safe_idx = clamp(brick_idx, vec3<i32>(0, 0, 0), lut_max);

    // Read brick -> cache tile mapping from the LUT.
    // textureLoad on texture_3d<u32> -> vec4<u32>.
    let lutv = textureLoad(t_lut, safe_idx, 0);

    // Reconstruct the cache coordinate.
    let tile_origin  = vec3<f32>(lutv.xyz) * block_size;
    let within_brick = pos - brick_f * block_size;
    let cache_pos    = tile_origin + within_brick;
    let cache_coord  = cache_pos / cache_size;

    return textureSample(t_img, s_img, cache_coord);
}


// Fragment stage - MIP raycast
// Replicates the full two-phase raycast from volume_ray.wgsl MIP mode:
//   1. Primary loop at relative_step_size intervals.
//   2. Secondary bisection refinement (4 iterations) for sub-step accuracy.
// sample_vol_lut() replaces the direct textureSample(t_img, ...) call.

@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    {$ include 'pygfx.clipping_planes.wgsl' $}

    let sizef = vec3<f32>(textureDimensions(t_img));
    let relative_step_size = clamp(
        sqrt(max(sizef.x, max(sizef.y, sizef.z))) / 20.0,
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
    let nsteps = i32(-dist / relative_step_size + 0.5);
    if (nsteps < 1) { discard; }
    let nstepsf = f32(nsteps);

    let start_coord = (front_pos + vec3<f32>(0.5, 0.5, 0.5)) / sizef;
    let step_coord  = ((back_pos - front_pos) / sizef) / f32(nsteps);

    // Primary loop: find approximate MIP location.
    var the_ref   = -999999.0;
    var the_coord = start_coord;
    var the_value : vec4<f32>;
    for (var iter = 0.0; iter < nstepsf; iter = iter + 1.0) {
        let coord = start_coord + iter * step_coord;
        let value = sample_vol_lut(coord, sizef);
        let reff = value.r;
        if (reff > the_ref) {
            the_ref   = reff;
            the_coord = coord;
            the_value = value;
        }
    }

    // Secondary refinement: 4 bisection iterations for sub-step accuracy.
    var substep_coord = step_coord;
    for (var iter2 = 0; iter2 < 4; iter2 = iter2 + 1) {
        substep_coord = substep_coord * 0.5;
        let coord1 = the_coord - substep_coord;
        let coord2 = the_coord + substep_coord;
        let value1 = sample_vol_lut(coord1, sizef);
        let value2 = sample_vol_lut(coord2, sizef);
        let ref1 = value1.r;
        let ref2 = value2.r;
        if (ref1 >= the_ref) {
            the_ref   = ref1;
            the_coord = coord1;
            the_value = value1;
        } else if (ref2 > the_ref) {
            the_ref   = ref2;
            the_coord = coord2;
            the_value = value2;
        }
    }

    // Colormapping.
    let color = sampled_value_to_color(the_value);
    // Move to physical colourspace (linear photon count).
    $$ if colorspace == 'srgb'
        let physical_color = srgb2physical(color.rgb);
    $$ else
        let physical_color = color.rgb;
    $$ endif
    let opacity = color.a * u_material.opacity;

    do_alpha_test(opacity);

    let data_pos  = the_coord * sizef - vec3<f32>(0.5, 0.5, 0.5);
    let world_pos = u_wobject.world_transform * vec4<f32>(data_pos, 1.0);
    let ndc_pos   = u_stdinfo.projection_transform
                  * u_stdinfo.cam_transform
                  * world_pos;

    var out: FragmentOutput;
    out.color = vec4<f32>(physical_color, opacity);
    out.depth = ndc_pos.z / ndc_pos.w;
    return out;
}
