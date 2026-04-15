# Opacity in Cellier v2

This note explains how opacity is controlled in Cellier v2, especially for 3D mesh overlays.

## Quick model

Opacity in the renderer is the combination of:

- `opacity`: alpha multiplier on the material color.
- `alpha_mode`: how fragments are composited (`solid` vs `blend`).
- `depth_test`: whether fragments are compared against the depth buffer.
- `depth_write`: whether fragments write depth.
- `side`: which mesh faces render (`front`, `back`, `both`).

For translucent objects, `opacity < 1` alone is not enough. You must also use blending and usually disable depth writes.

## Mesh behavior in v2

3D mesh visuals (`src/cellier/v2/render/visuals/_mesh_memory.py`) now apply explicit transparency state:

- If `opacity < 1.0`:
  - `alpha_mode = "blend"`
  - `depth_test = True`
  - `depth_write = False`
- If `opacity >= 1.0`:
  - `alpha_mode = "solid"`
  - `depth_test = True`
  - `depth_write = True`

This policy is applied:

- when 3D mesh materials are created, and
- when opacity changes at runtime (`on_appearance_changed`).

Why: this allows seeing image/volume content through a translucent mesh while keeping correct front-of-scene depth testing.

## Side (`front` vs `both`)

For transparent shells, `side` changes the look significantly:

- `side="front"`: cleaner overlay, usually preferred for translucent boxes.
- `side="both"`: draws front and back faces, often appears denser/darker due to double contribution.

Use `both` if you intentionally want interior/backface visibility; otherwise `front` is typically easier to read.

## Current transform validation viewer setup

In `scripts/v2/transform_validation_viewer.py`, the 3D mesh uses:

- `opacity=0.8`
- `side="front"`

This combination is intended to keep the volume visible through the mesh while still showing a clear geometric envelope.

## Practical guidance

1. Start with `opacity` in the `0.6` to `0.85` range for overlays.
2. Keep transparent meshes in blended mode with `depth_write=False`.
3. Prefer `side="front"` for clean inspection overlays.
4. Switch to `side="both"` only when you need backfaces for debugging.

## Troubleshooting

If a translucent mesh still hides content behind it:

- verify `opacity < 1`,
- verify `alpha_mode` is `blend`,
- verify `depth_write` is `False`, and
- check whether `side="both"` is making the shell look unintentionally opaque.

## References

- `src/cellier/v2/render/visuals/_mesh_memory.py`
- `src/cellier/v2/visuals/_mesh_memory.py`
- `tests/v2/render/test_gfx_mesh_memory_visual.py`
- `scripts/v2/transform_validation_viewer.py`

