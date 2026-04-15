# 2D proxy texture sizing and LUT lookup

This document explains how 2D multiscale image rendering works in Cellier v2
after the proxy/LUT alignment fixes.

The key idea is:

- geometry footprint uses the true image size `(W, H)` at the finest level, and
- shader lookup uses true voxel-space coordinates derived from that same
  `(W, H)` domain.

This keeps on-screen geometry and cache sampling in the same coordinate system.

## Architecture overview

The 2D path has three layers:

1. **Geometry layer** (`GFXMultiscaleImageVisual._build_2d_node`)
   - creates a proxy texture for pygfx image geometry,
   - scales the image node so displayed footprint matches true `(W, H)`.
2. **Cache/LUT layer** (`BlockCache2D`, `LutIndirectionManager2D`)
   - stores tiles in a fixed-size cache atlas,
   - stores per-tile indirection in a LUT texture.
3. **Shader layer** (`image_block.wgsl`)
   - maps fragment `texcoord` to voxel-space position,
   - resolves tile via LUT,
   - samples from cache atlas.

## Axis conventions

Two axis conventions are used:

| Domain | Order |
|---|---|
| Data/numpy | `(H, W)` = `(y, x)` |
| Shader/pygfx | `(x, y)` = `(W, H)` |

Whenever values cross CPU data code to shader code, axes are reordered to shader
order.

## Proxy texture and true footprint

In `_build_2d_node` (`src/cellier/v2/render/visuals/_image.py`):

- proxy texture shape is tile grid: `(gH, gW)`;
- base tile size is `block_size = bs`;
- true finest-level shape is `(H, W)`.

If shapes are not exact tile multiples, padded proxy extent is:

- width: `gW * bs`
- height: `gH * bs`

Displayed geometry is corrected to true extent with per-axis factors:

- `sx = W / (gW * bs)`
- `sy = H / (gH * bs)`

Then local node transform uses:

- `scale_x = bs * sx`
- `scale_y = bs * sy`

So the rendered quad footprint is exactly `(W, H)` (in data-space units before
world transform), not padded `(gW * bs, gH * bs)`.

## LUT parameter semantics (2D)

`build_lut_params_buffer_2d` (`src/cellier/v2/render/lut_indirection/_lut_buffers_2d.py`)
now emits voxel-space semantics for shader sampling:

| Uniform field | Meaning |
|---|---|
| `block_size_x/y` | tile size in voxels (`bs`) |
| `overlap` | overlap border in voxels |
| `cache_size_x/y` | cache atlas extent in voxels |
| `lut_size_x/y` | LUT grid size `(gW, gH)` |
| `vol_size_x/y` | true image size `(W, H)` |

Important: `vol_size_*` is true image size, not padded tile-rounded size.

## Shader lookup flow in voxel space

`sample_im_lut` in `src/cellier/v2/render/shaders/wgsl/image_block.wgsl`
operates as follows.

1. Convert fragment texcoord to voxel-space position:

   - `pos = clamp(texcoord * vol_size, 0, vol_size - 0.5)`

2. Resolve tile index:

   - `tile_f = floor(pos / block_size)`
   - `tile_idx = clamp(tile_f, 0, lut_size - 1)`

3. Read LUT entry:

   - `lutv = textureLoad(t_lut, tile_idx, 0)`
   - `lutv.xy` = cache slot coords
   - `lutv.z` = level (1-based, `0` means unloaded)

4. Compute cache sample position:

   - `tile_origin = lutv.xy * padded_size`
   - `scaled_pos = pos * get_tile_scale(level)`
   - `within_tile = scaled_pos - floor(scaled_pos / block_size) * block_size`
   - `cache_pos = tile_origin + within_tile + overlap`
   - `cache_coord = cache_pos / cache_size`

5. Sample atlas:

   - `textureSample(t_cache, s_cache, cache_coord)`

## LOD convention and fallback

2D LUT uses 1-based level encoding:

- `0`: tile missing (render black/transparent)
- `1`: finest
- `2+`: coarser levels

LUT rebuild writes coarsest-to-finest so fine tiles overwrite coarse where
available, with automatic coarse fallback when fine data is missing.

## Why this fixed the distortion

Before these changes, geometry and lookup used different effective domains:

- geometry was being corrected to true `(W, H)`, but
- lookup still behaved like proxy-grid/padded space.

That mismatch caused stretch and offset artifacts (especially on non-square or
non-multiple-of-block dimensions such as XZ/YZ in transform-validation data).

Now both geometry and lookup share true voxel-space `(W, H)` semantics, so
aspect and positioning are consistent.

## Practical debugging checklist

If 2D tiles look stretched or shifted:

1. verify `_build_2d_node` correction is active (`sx`, `sy` from `(W,H)` vs `(gW,gH)`),
2. verify `u_lut_params.vol_size_* == (W, H)`,
3. verify `u_lut_params.block_size_* == bs` in voxel units,
4. verify WGSL uses `texcoord * vol_size` (not proxy texture dimensions),
5. verify axis order is `(x=W, y=H)` in shader-facing values.

## Related files

- `src/cellier/v2/render/visuals/_image.py`
- `src/cellier/v2/render/lut_indirection/_lut_buffers_2d.py`
- `src/cellier/v2/render/lut_indirection/_layout_2d.py`
- `src/cellier/v2/render/lut_indirection/_lut_indirection_manager_2d.py`
- `src/cellier/v2/render/shaders/_block_image.py`
- `src/cellier/v2/render/shaders/wgsl/image_block.wgsl`
- `scripts/v2/transform_validation_viewer.py`

