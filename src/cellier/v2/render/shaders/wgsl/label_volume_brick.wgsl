{$ include 'pygfx.std.wgsl' $}
// NOTE: do NOT include pygfx colormap, image_sample, or volume_common.
// Labels use textureLoad (nearest-neighbor on integer textures), not textureSample.

// ── Constants ──────────────────────────────────────────────────────────────
const BORDER: f32          = 1.0;     // ghost border (nearest-neighbor only; no gradient)
const MAX_BRICK_ITERS: u32 = 512u;
const STEPS_PER_BRICK: f32 = 24.0;

// NOTE: Do NOT define struct LabelParams, VOL_PARAMS, or BlockScales here.
// pygfx auto-generates them from numpy dtypes via structname=.

// ── Block-scale lookup ────────────────────────────────────────────────────
fn get_lod_scale(lut_w: u32) -> vec3<f32> {
    switch lut_w {
        case 1u:  { return u_block_scales.scale_1.xyz; }
        case 2u:  { return u_block_scales.scale_2.xyz; }
        case 3u:  { return u_block_scales.scale_3.xyz; }
        case 4u:  { return u_block_scales.scale_4.xyz; }
        case 5u:  { return u_block_scales.scale_5.xyz; }
        case 6u:  { return u_block_scales.scale_6.xyz; }
        case 7u:  { return u_block_scales.scale_7.xyz; }
        case 8u:  { return u_block_scales.scale_8.xyz; }
        case 9u:  { return u_block_scales.scale_9.xyz; }
        default:  { return u_block_scales.scale_1.xyz; }
    }
}

// ── Coordinate conversions ────────────────────────────────────────────────

fn norm_to_voxel(pos: vec3<f32>, norm_size: vec3<f32>, dataset_size: vec3<f32>) -> vec3<f32> {
    return ((pos / norm_size) + 0.5) * dataset_size;
}

fn voxel_to_norm(voxel: vec3<f32>, norm_size: vec3<f32>, dataset_size: vec3<f32>) -> vec3<f32> {
    return (voxel / dataset_size - 0.5) * norm_size;
}

// ── Ray-box intersection ──────────────────────────────────────────────────

fn intersect_box(
    ray_origin: vec3<f32>, inv_ray_dir: vec3<f32>,
    box_min: vec3<f32>,   box_max: vec3<f32>
) -> vec2<f32> {
    let t0 = (box_min - ray_origin) * inv_ray_dir;
    let t1 = (box_max - ray_origin) * inv_ray_dir;
    let t_near = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let t_far  = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    return vec2<f32>(t_near, t_far);
}

// ── HSV → RGB ──────────────────────────────────────────────────────────────
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let hp = h * 6.0;
    let i  = floor(hp);
    let f  = hp - i;
    let p  = v * (1.0 - s);
    let q  = v * (1.0 - s * f);
    let t  = v * (1.0 - s * (1.0 - f));
    switch (i32(i) % 6) {
        case 0:  { return vec3<f32>(v, t, p); }
        case 1:  { return vec3<f32>(q, v, p); }
        case 2:  { return vec3<f32>(p, v, t); }
        case 3:  { return vec3<f32>(p, q, v); }
        case 4:  { return vec3<f32>(t, p, v); }
        default: { return vec3<f32>(v, p, q); }
    }
}

// ── Random colormap ────────────────────────────────────────────────────────
fn random_label_color(label_id: i32, salt: u32) -> vec4<f32> {
    var x = bitcast<u32>(label_id) ^ salt;
    x = (x ^ (x >> 16u)) * 0x45d9f3bu;
    x = (x ^ (x >> 16u)) * 0x45d9f3bu;
    x = x ^ (x >> 16u);
    let hue  = f32(x & 0xFFFFu) / 65535.0;
    let sat  = 0.6 + 0.4 * f32((x >> 16u) & 0x3Fu) / 63.0;
    let val_ = 0.7 + 0.3 * f32((x >> 22u) & 0xFu)  / 15.0;
    return vec4<f32>(hsv_to_rgb(hue, sat, val_), 1.0);
}

// ── Direct colormap (only compiled in direct mode) ────────────────────────
$$ if colormap_mode == "direct"
fn direct_label_color(label_id: i32, n: i32) -> vec4<f32> {
    var lo = 0; var hi = n;
    for (var iter = 0; iter < 17; iter++) {
        if (lo >= hi) { break; }
        let mid = (lo + hi) / 2;
        let key = textureLoad(t_label_keys, vec2<i32>(mid, 0), 0).r;
        if (key == label_id) {
            return textureLoad(t_label_colors, vec2<i32>(mid, 0), 0);
        }
        if (key < label_id) { lo = mid + 1; } else { hi = mid; }
    }
    return vec4<f32>(0.0);
}
$$ endif

fn get_label_color(label_id: i32) -> vec4<f32> {
    $$ if colormap_mode == "random"
    return random_label_color(label_id, u_label_params.salt);
    $$ else
    return direct_label_color(label_id, i32(u_label_params.n_entries));
    $$ endif
}

// ── Atlas sampler (integer, nearest-neighbor) ─────────────────────────────
fn sample_atlas_label(
    voxel_pos:      vec3<f32>,
    lut_entry:      vec4<u32>,
    lod_scale:      vec3<f32>,
    brick_corner_k: vec3<f32>,
) -> i32 {
    let block_size  = vec3<f32>(u_vol_params.block_size_x,
                                u_vol_params.block_size_y,
                                u_vol_params.block_size_z);
    let cache_size  = vec3<i32>(i32(u_vol_params.cache_size_x),
                                i32(u_vol_params.cache_size_y),
                                i32(u_vol_params.cache_size_z));
    let padded_size = block_size + vec3<f32>(2.0 * BORDER);

    let tile_origin  = vec3<f32>(lut_entry.xyz) * padded_size;
    let voxel_k      = voxel_pos / lod_scale;
    let pos_in_brick = voxel_k - brick_corner_k;

    // Nearest-neighbor: round, no +0.5 sub-voxel shift.
    let cache_pos = tile_origin + pos_in_brick + vec3<f32>(BORDER);
    let texel     = clamp(vec3<i32>(round(cache_pos)), vec3<i32>(0), cache_size - vec3<i32>(1));
    return textureLoad(t_cache, texel, 0).r;
}

// ── Brick setup ───────────────────────────────────────────────────────────

struct BrickInfo {
    lut_entry:      vec4<u32>,
    lod_scale:      vec3<f32>,
    brick_corner_k: vec3<f32>,
    t_end:          f32,
    num_steps:      u32,
    step_size:      f32,
    brick_max:      f32,
    valid:          bool,
};

fn setup_brick(
    ray_origin:  vec3<f32>,
    ray_dir:     vec3<f32>,
    inv_ray_dir: vec3<f32>,
    t:           f32,
    t_end_ray:   f32,
    norm_size:   vec3<f32>,
    dataset_size: vec3<f32>,
) -> BrickInfo {
    var info: BrickInfo;

    let block_size = vec3<f32>(u_vol_params.block_size_x,
                               u_vol_params.block_size_y,
                               u_vol_params.block_size_z);
    let lut_size_i = vec3<i32>(i32(u_vol_params.lut_size_x),
                               i32(u_vol_params.lut_size_y),
                               i32(u_vol_params.lut_size_z));

    let sample_pos = ray_origin + ray_dir * t;
    let voxel_pos  = clamp(
        norm_to_voxel(sample_pos, norm_size, dataset_size),
        vec3<f32>(0.0),
        dataset_size - vec3<f32>(0.001)
    );
    let brick_f   = floor(voxel_pos / block_size);
    let safe_idx  = clamp(vec3<i32>(brick_f),
                          vec3<i32>(0),
                          lut_size_i - vec3<i32>(1));

    let brick_min_v = brick_f * block_size;
    let brick_max_v = brick_min_v + block_size;
    let brick_min_n = voxel_to_norm(brick_min_v, norm_size, dataset_size);
    let brick_max_n = voxel_to_norm(brick_max_v, norm_size, dataset_size);

    let t_brick    = intersect_box(ray_origin, inv_ray_dir, brick_min_n, brick_max_n);
    info.t_end     = min(t_brick.y, t_end_ray);

    let lut_entry  = textureLoad(t_lut, safe_idx, 0);
    info.lut_entry = lut_entry;
    info.brick_max = textureLoad(t_brick_max, safe_idx, 0).r;
    info.valid     = lut_entry.w > 0u;

    if (info.valid) {
        info.lod_scale      = get_lod_scale(lut_entry.w);
        let voxel_k_bc      = voxel_pos / info.lod_scale;
        info.brick_corner_k = floor(voxel_k_bc / block_size) * block_size;
        let brick_len       = max(info.t_end - t, 1e-6);
        let brick_world_len = length(brick_max_n - brick_min_n);
        let max_scale  = max(info.lod_scale.x, max(info.lod_scale.y, info.lod_scale.z));
        info.num_steps = max(1u, u32((STEPS_PER_BRICK / max_scale) * brick_len / brick_world_len));
        info.step_size = brick_len / f32(info.num_steps);
    }

    return info;
}

// ── Vertex shader ─────────────────────────────────────────────────────────

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
};

@vertex
fn vs_main(in: VertexInput) -> Varyings {
    let ns = vec3<f32>(u_vol_params.norm_size_x,
                       u_vol_params.norm_size_y,
                       u_vol_params.norm_size_z);
    let h = ns * 0.5;

    let corners = array<vec3<f32>, 8>(
        vec3<f32>(-h.x, -h.y, -h.z),
        vec3<f32>( h.x, -h.y, -h.z),
        vec3<f32>(-h.x,  h.y, -h.z),
        vec3<f32>( h.x,  h.y, -h.z),
        vec3<f32>(-h.x, -h.y,  h.z),
        vec3<f32>( h.x, -h.y,  h.z),
        vec3<f32>(-h.x,  h.y,  h.z),
        vec3<f32>( h.x,  h.y,  h.z),
    );

    let indices = array<u32, 36>(
        0u,2u,1u, 1u,2u,3u,
        4u,5u,6u, 5u,7u,6u,
        0u,1u,4u, 1u,5u,4u,
        2u,6u,3u, 3u,6u,7u,
        0u,4u,2u, 2u,4u,6u,
        1u,3u,5u, 3u,7u,5u,
    );

    let norm_pos  = corners[indices[in.vertex_index]];
    let world_pos = u_wobject.world_transform * vec4<f32>(norm_pos, 1.0);
    let ndc_pos   = u_stdinfo.projection_transform
                  * u_stdinfo.cam_transform
                  * world_pos;

    let ndc_to_norm = u_wobject.world_transform_inv
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
    varyings.norm_near_pos = vec4<f32>(ndc_to_norm * vec4<f32>(ndc_pos.xy, -1.0 * cam_sign * ndc_pos.w, ndc_pos.w));
    varyings.norm_far_pos  = vec4<f32>(ndc_to_norm * vec4<f32>(ndc_pos.xy,        cam_sign * ndc_pos.w, ndc_pos.w));
    return varyings;
}


// ── Fragment shader ───────────────────────────────────────────────────────

@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    {$ include 'pygfx.clipping_planes.wgsl' $}

    var out: FragmentOutput;

    let norm_size    = vec3<f32>(u_vol_params.norm_size_x,
                                 u_vol_params.norm_size_y,
                                 u_vol_params.norm_size_z);
    let dataset_size = vec3<f32>(u_vol_params.dataset_size_x,
                                 u_vol_params.dataset_size_y,
                                 u_vol_params.dataset_size_z);
    let background_label = u_label_params.background_label;

    // ── Ray setup ──────────────────────────────────────────────────────────
    let far_pos  = varyings.norm_far_pos.xyz  / varyings.norm_far_pos.w;
    let near_pos = varyings.norm_near_pos.xyz / varyings.norm_near_pos.w;
    let ray_dir  = normalize(far_pos - near_pos);

    let safe_dir    = select(ray_dir, vec3<f32>(1e-10), abs(ray_dir) < vec3<f32>(1e-10));
    let inv_ray_dir = 1.0 / safe_dir;

    let half_ns = norm_size * 0.5;
    let hit     = intersect_box(near_pos, inv_ray_dir, -half_ns, half_ns);
    if (hit.x > hit.y) { discard; }
    var t_start = max(hit.x, 0.0);
    let t_end   = hit.y;
    if (t_start >= t_end) { discard; }

    let ray_origin = near_pos;

    var t             = t_start;
    var surface_found = false;
    // surface_pos: midpoint after bisection (used for depth).
    var surface_pos   = vec3<f32>(0.0);
    // surface_hi_p: last confirmed-foreground bisection position (use for color).
    // Using surface_pos (midpoint) for label ID can snap to background and produce
    // incorrect colors. Always use surface_hi_p for get_label_color().
    var surface_hi_p  = vec3<f32>(0.0);
    var surface_lut_entry: vec4<u32>;
    var surface_lod_scale: vec3<f32>;
    var surface_brick_corner_k: vec3<f32>;
    var prev_t: f32 = t_start;
    var prev_lut_entry:      vec4<u32>;
    var prev_lod_scale:      vec3<f32>;
    var prev_brick_corner_k: vec3<f32>;
    var brick_boundary_t: f32 = t_start;

    for (var brick_iter = 0u; brick_iter < MAX_BRICK_ITERS; brick_iter++) {
        if (t >= t_end || surface_found) { break; }

        let brick = setup_brick(ray_origin, ray_dir, inv_ray_dir,
                                t, t_end, norm_size, dataset_size);

        brick_boundary_t = t;

        if (!brick.valid) {
            t = brick.t_end + 0.0001;
            prev_t = t;
            continue;
        }

        // Skip bricks containing no labels (brick_max == 0.0 means all-background).
        if (brick.brick_max < 0.5) {
            t = brick.t_end + 0.0001;
            prev_t = t;
            continue;
        }

        surface_lut_entry      = brick.lut_entry;
        surface_lod_scale      = brick.lod_scale;
        surface_brick_corner_k = brick.brick_corner_k;

        var t_sample = t;
        for (var i = 0u; i < brick.num_steps; i++) {
            if (t_sample > brick.t_end) { break; }

            let pos    = ray_origin + ray_dir * t_sample;
            let voxel  = clamp(
                norm_to_voxel(pos, norm_size, dataset_size),
                vec3<f32>(0.0),
                dataset_size - vec3<f32>(0.001)
            );
            let lid = sample_atlas_label(voxel, brick.lut_entry,
                                         brick.lod_scale, brick.brick_corner_k);

            if (lid != background_label) {
                // 10-step bisection for sub-voxel surface refinement.
                // norm space: lo_p = previous sample position, hi_p = current.
                var lo_p_norm = ray_origin + ray_dir * prev_t;
                var hi_p_norm = pos;
                for (var r = 0; r < 10; r++) {
                    let mid_norm  = (lo_p_norm + hi_p_norm) * 0.5;
                    let mid_voxel = clamp(
                        norm_to_voxel(mid_norm, norm_size, dataset_size),
                        vec3<f32>(0.0),
                        dataset_size - vec3<f32>(0.001)
                    );
                    let mid_id = sample_atlas_label(
                        mid_voxel,
                        surface_lut_entry,
                        surface_lod_scale,
                        surface_brick_corner_k,
                    );
                    if (mid_id != background_label) {
                        hi_p_norm = mid_norm;
                    } else {
                        lo_p_norm = mid_norm;
                    }
                }
                surface_pos   = (lo_p_norm + hi_p_norm) * 0.5;
                surface_hi_p  = hi_p_norm;
                surface_found = true;
                break;
            }
            prev_t  = t_sample;
            prev_lut_entry      = brick.lut_entry;
            prev_lod_scale      = brick.lod_scale;
            prev_brick_corner_k = brick.brick_corner_k;
            t_sample += brick.step_size;
        }
        if (!surface_found) {
            t = brick.t_end + 0.0001;
        }
    }

    // Hoist world_surface before discards so dpdx/dpdy see defined derivatives
    // across the full 2×2 quad. Pixels that find no surface use the exit point
    // as a smooth fallback and are discarded below.
    let surface_or_exit = select(
        ray_origin + ray_dir * t_end,
        surface_pos,
        surface_found
    );
    let world_surface = u_wobject.world_transform * vec4<f32>(surface_or_exit, 1.0);

    $$ if render_mode == "iso_categorical"
    // Screen-space normal from world-position derivatives across the 2×2 quad.
    // cross(dpdy, dpdx) points toward the camera in y-down screen convention.
    let ss_normal = cross(dpdy(world_surface.xyz), dpdx(world_surface.xyz));
    $$ endif

    // Deferred discards (must come after derivative calls).
    if (!surface_found) { discard; }

    // Sample label identity from surface_hi_p (guaranteed foreground voxel).
    let hi_voxel = clamp(
        norm_to_voxel(surface_hi_p, norm_size, dataset_size),
        vec3<f32>(0.0),
        dataset_size - vec3<f32>(0.001)
    );
    let lid_surface = sample_atlas_label(
        hi_voxel,
        surface_lut_entry,
        surface_lod_scale,
        surface_brick_corner_k,
    );
    let color = get_label_color(lid_surface);
    if (color.a < 0.001) { discard; }

    $$ if render_mode == "iso_categorical"
    let view_dir = normalize(-ray_dir);
    let normal   = select(
        normalize(ss_normal),
        vec3<f32>(0.0, 1.0, 0.0),
        dot(ss_normal, ss_normal) < 1e-10,
    );
    let N        = select(-normal, normal, dot(normal, view_dir) > 0.0);
    let ambient  = 0.3;
    let diffuse  = max(dot(N, view_dir), 0.0);
    let lit_color = color.rgb * (ambient + diffuse);
    $$ else
    let lit_color = color.rgb;
    $$ endif

    let ndc_surface = u_stdinfo.projection_transform
                    * u_stdinfo.cam_transform
                    * world_surface;

    out.color = vec4<f32>(lit_color, color.a * u_material.opacity);
    out.depth = ndc_surface.z / ndc_surface.w;
    return out;
}
