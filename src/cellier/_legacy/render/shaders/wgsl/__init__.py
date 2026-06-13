"""Custom WGSL shaders for pygfx."""

import functools
import importlib.resources


@functools.lru_cache(maxsize=None)
def load_wgsl(shader_name, package_name: str = "cellier.render.shaders.wgsl"):
    """Load wgsl code from pygfx builtin shader snippets.

    from: pygfx/renderers/wgpu/wgsl/__init__.py
    """
    ref = importlib.resources.files(package_name) / shader_name
    context = importlib.resources.as_file(ref)
    with context as path:
        with open(path, "rb") as f:
            return f.read().decode()
