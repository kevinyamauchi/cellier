{$ include 'pygfx.std.wgsl' $}

// ── Custom bindings ───────────────────────────────────────────────────────
// t_img          : texture_3d<i32>   — full label volume (r32sint)
// t_label_keys   : texture_2d<i32>   — direct-mode sorted keys
// t_label_colors : texture_2d<f32>   — direct-mode RGBA
// u_label_params : LabelParams uniform (background_label, salt, n_entries, _pad)
//
// NOTE: Do NOT define struct LabelParams here.
// pygfx auto-generates it from the numpy dtype via structname=.
//
// Local (object) space convention: voxel i is centered at i.
// The volume of shape (W, H, D) spans [-0.5, W-0.5] x [-0.5, H-0.5] x [-0.5, D-0.5].
// world_transform maps local space to world space.

// ── HSV → RGB ─────────────────────────────────────────────────────────────
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

// ── Nearest-neighbor label sample (local voxel space) ────────────────────
fn sample_label(local_pos: vec3<f32>, vol_dims: vec3<i32>) -> i32 {
    let texel = clamp(
        vec3<i32>(round(local_pos)),
        vec3<i32>(0),
        vol_dims - vec3<i32>(1),
    );
    return textureLoad(t_img, texel, 0).r;
}

// ── Binary foreground mask ────────────────────────────────────────────────
fn foreground(label_id: i32) -> f32 {
    return select(0.0, 1.0, label_id != u_label_params.background_label);
}

// ── Ray-AABB intersection (slab test) ────────────────────────────────────
fn intersect_aabb(
    ray_o: vec3<f32>, inv_d: vec3<f32>,
    box_min: vec3<f32>, box_max: vec3<f32>,
) -> vec2<f32> {
    let t0 = (box_min - ray_o) * inv_d;
    let t1 = (box_max - ray_o) * inv_d;
    let t_near = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let t_far  = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    return vec2<f32>(t_near, t_far);
}

// ── Vertex shader ─────────────────────────────────────────────────────────
struct VertexInput { @builtin(vertex_index) vertex_index: u32 };

@vertex
fn vs_main(in: VertexInput) -> Varyings {
    // Build bounding-box corners from texture dimensions.
    // pygfx voxel convention: voxel i is centered at i; box spans [-0.5, N-0.5].
    let dims = vec3<f32>(textureDimensions(t_img));
    let lo   = vec3<f32>(-0.5);
    let hi   = dims - vec3<f32>(0.5);

    let corners = array<vec3<f32>, 8>(
        vec3<f32>(lo.x, lo.y, lo.z),
        vec3<f32>(hi.x, lo.y, lo.z),
        vec3<f32>(lo.x, hi.y, lo.z),
        vec3<f32>(hi.x, hi.y, lo.z),
        vec3<f32>(lo.x, lo.y, hi.z),
        vec3<f32>(hi.x, lo.y, hi.z),
        vec3<f32>(lo.x, hi.y, hi.z),
        vec3<f32>(hi.x, hi.y, hi.z),
    );
    // 12 triangles (36 indices), cull_mode = none.
    let face_idx = array<u32, 36>(
        0u,2u,1u, 1u,2u,3u,
        4u,5u,6u, 5u,7u,6u,
        0u,1u,4u, 1u,5u,4u,
        2u,6u,3u, 3u,6u,7u,
        0u,4u,2u, 2u,4u,6u,
        1u,3u,5u, 3u,7u,5u,
    );

    let local_pos = corners[face_idx[in.vertex_index]];
    let world_pos = u_wobject.world_transform * vec4<f32>(local_pos, 1.0);
    let ndc_pos   = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

    // Pass near/far clip-plane positions in local space for ray reconstruction.
    let ndc_to_local  = u_wobject.world_transform_inv * u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv;
    let cam_sign = sign(
        u_stdinfo.cam_transform[0][0] *
        u_stdinfo.cam_transform[1][1] *
        u_stdinfo.cam_transform[2][2]
    );

    var varyings: Varyings;
    varyings.position      = vec4<f32>(ndc_pos);
    varyings.world_pos     = vec3<f32>(world_pos.xyz);
    varyings.local_near    = vec4<f32>(ndc_to_local * vec4<f32>(ndc_pos.xy, -1.0 * cam_sign * ndc_pos.w, ndc_pos.w));
    varyings.local_far     = vec4<f32>(ndc_to_local * vec4<f32>(ndc_pos.xy,        cam_sign * ndc_pos.w, ndc_pos.w));
    return varyings;
}

// ── Fragment shader ───────────────────────────────────────────────────────
@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    let vol_dims  = vec3<i32>(textureDimensions(t_img));
    let vol_dimsf = vec3<f32>(vol_dims);

    // Reconstruct ray in local (voxel) space.
    let near_pos = varyings.local_near.xyz / varyings.local_near.w;
    let far_pos  = varyings.local_far.xyz  / varyings.local_far.w;
    let ray_dir  = normalize(far_pos - near_pos);

    // Ray-AABB intersection with local box [-0.5, dims-0.5].
    let lo = vec3<f32>(-0.5);
    let hi = vol_dimsf - vec3<f32>(0.5);
    let safe_dir    = select(ray_dir, vec3<f32>(1e-20), abs(ray_dir) < vec3<f32>(1e-20));
    let inv_dir     = 1.0 / safe_dir;
    let t_hit       = intersect_aabb(near_pos, inv_dir, lo, hi);
    if (t_hit.x > t_hit.y) { discard; }
    let t_start = max(t_hit.x, 0.0);
    let t_end   = t_hit.y;

    // Step in voxel units (0.8 voxels for reliable detection of thin labels).
    let step_size = 0.8;
    let n_steps   = i32(ceil((t_end - t_start) / step_size)) + 1;

    var surface_found = false;
    var surface_pos   = vec3<f32>(0.0);
    // hi_p is the last confirmed-foreground bisection position; used for label
    // identity sampling to avoid the midpoint rounding into the background voxel.
    var surface_hi_p  = vec3<f32>(0.0);
    var prev_pos      = near_pos + ray_dir * t_start;

    for (var i = 0; i < n_steps; i++) {
        let t   = t_start + f32(i) * step_size;
        if (t > t_end) { break; }
        let pos = near_pos + ray_dir * t;
        let lid = sample_label(pos, vol_dims);
        if (lid != u_label_params.background_label) {
            // 10-step bisection for sub-voxel surface refinement.
            var lo_p = prev_pos;
            var hi_p = pos;
            for (var r = 0; r < 10; r++) {
                let mid_p  = (lo_p + hi_p) * 0.5;
                let mid_id = sample_label(mid_p, vol_dims);
                if (mid_id != u_label_params.background_label) {
                    hi_p = mid_p;
                } else {
                    lo_p = mid_p;
                }
            }
            surface_pos   = (lo_p + hi_p) * 0.5;
            surface_hi_p  = hi_p;
            surface_found = true;
            break;
        }
        prev_pos = pos;
    }

    // Hoist world_surface before discards so dpdx/dpdy see a defined value
    // across the full 2x2 quad. Pixels with no surface use the exit point as
    // a smooth fallback and are discarded below as helper invocations.
    let surface_or_exit = select(near_pos + ray_dir * t_end, surface_pos, surface_found);
    let world_surface   = u_wobject.world_transform * vec4<f32>(surface_or_exit, 1.0);

    $$ if render_mode == "iso_categorical"
    // Screen-space normal from world-position derivatives across the 2x2 quad.
    // cross(dpdy, dpdx) points toward the camera in y-down screen convention.
    let ss_normal = cross(dpdy(world_surface.xyz), dpdx(world_surface.xyz));
    $$ endif

    // Deferred discards.
    if (!surface_found) { discard; }

    // Sample label identity from surface_hi_p (guaranteed foreground).
    // surface_pos is the midpoint which straddles the boundary; round() on
    // it can snap to the background voxel, producing red/magenta fringes.
    let lid_surface = sample_label(surface_hi_p, vol_dims);
    let color       = get_label_color(lid_surface);
    if (color.a < 0.001) { discard; }

    $$ if render_mode == "iso_categorical"
    let view_dir  = normalize(-ray_dir);
    let normal    = select(
        normalize(ss_normal),
        vec3<f32>(0.0, 1.0, 0.0),
        dot(ss_normal, ss_normal) < 1e-10,
    );
    let N         = select(-normal, normal, dot(normal, view_dir) > 0.0);
    let ambient   = 0.3;
    let diffuse   = max(dot(N, view_dir), 0.0);
    let lit_color = color.rgb * (ambient + diffuse);
    $$ else
    let lit_color = color.rgb;
    $$ endif

    let ndc_surface = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_surface;

    var out: FragmentOutput;
    out.color = vec4<f32>(lit_color, color.a * u_material.opacity);
    out.depth = ndc_surface.z / ndc_surface.w;
    return out;
}
