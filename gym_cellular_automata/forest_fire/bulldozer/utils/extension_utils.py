"""Extension and transformation utilities for the Advanced Bulldozer environment."""

import jax
import jax.numpy as jnp
import flax
import numpy as np

from typing import Callable
from functools import partial


@jax.jit
def wind_fn(grid, per_env_context, shared_context, skip_visibility=0, skip_blur=0):
    """Calculate wind gradient based on wind direction.

    Args:
        grid: Environment grid
        per_env_context: Per-environment context containing wind_index
        shared_context: Shared context parameters
        skip_visibility: Flag to skip visibility transform
        skip_blur: Flag to skip blur transform

    Returns:
        Wind gradient array increasing in wind direction
    """
    # Get wind probabilities for each environment
    wind_index = per_env_context["wind_index"]

    # Create coordinate grids
    x, y = jnp.meshgrid(
        jnp.linspace(0, 1, grid.shape[1]),  # width dimension
        jnp.linspace(0, 1, grid.shape[0]),  # height dimension
    )

    # Convert wind index to angle (in radians)
    # 0 (North) = π/2, then rotate clockwise
    angle = (jnp.pi / 2) - (wind_index * jnp.pi / 4)

    # Calculate directional components
    x_component = jnp.cos(angle)
    y_component = jnp.sin(angle)

    # Create gradient that increases in wind direction
    wind_gradient = y * y_component + x * x_component

    return wind_gradient


@jax.jit
def density_fn(grid, per_env_context, shared_context, skip_visibility=0, skip_blur=0):
    """Return density information from context."""
    return per_env_context["density"]


@jax.jit
def vegetation_fn(
    grid, per_env_context, shared_context, skip_visibility=0, skip_blur=0
):
    """Return vegetation information from context."""
    return per_env_context["vegetation"]


@jax.jit
def altitude_fn(grid, per_env_context, shared_context, skip_visibility=0, skip_blur=0):
    """Return altitude information from context."""
    return per_env_context["altitude"]


@jax.jit
def noop_fn(grid, per_env_context, shared_context, skip_visibility=0, skip_blur=0):
    """Return zero array of same shape as grid."""
    return jnp.zeros_like(grid)


@jax.jit
def unblur_fn(grid, per_env_context, shared_context, skip_visibility=0, skip_blur=0):
    """Show grid with specified transformation skips."""
    return grid


@jax.jit
def see_invisible_fires_fn(
    grid, per_env_context, shared_context, skip_visibility=1, skip_blur=0
):
    """Show grid with specified transformation skips."""
    return grid


@jax.jit
def apply_visibility(grid, per_env_context):
    """Hide fires during daytime."""
    return jnp.where(
        (grid == 25) & (per_env_context["is_night"] == 0),
        0,  # Hide fires during day
        grid,
    )


@jax.jit
def apply_blur(grid):
    """Apply uniform blur transformation."""
    # Normalize values
    normalized = jnp.where(grid == 0, 0.0, jnp.where(grid == 3, 0.5, 1.0))

    # Apply uniform blur
    kernel = jnp.ones((3, 3)) / 9.0
    padded = jnp.pad(normalized, ((1, 1), (1, 1)), mode="edge")
    blurred = jnp.zeros_like(normalized)
    for i in range(3):
        for j in range(3):
            blurred += (
                kernel[i, j] * padded[i : i + grid.shape[0], j : j + grid.shape[1]]
            )

    # Map back to original values
    return jnp.where(blurred < 1 / 3, 0, jnp.where(blurred < 2 / 3, 3, 25))


@jax.jit
def transform_grid(grid, per_env_context, skip_visibility, skip_blur):
    """Apply sequence of transformations, conditionally based on skip flags.

    Args:
        grid: Environment grid to transform
        per_env_context: Per-environment context
        skip_visibility: Whether to skip visibility transform (0 or 1)
        skip_blur: Whether to skip blur transform (0 or 1)

    Returns:
        Transformed grid
    """
    grid = jnp.where(skip_visibility, grid, apply_visibility(grid, per_env_context))
    grid = jnp.where(skip_blur, grid, apply_blur(grid))
    return grid


@jax.jit
def apply_extension_fn(fn_idx, transformed_grid, per_env_context, shared_context):
    """Apply a specific extension function based on index."""
    return jax.lax.switch(
        fn_idx,
        [
            partial(fn, transformed_grid, per_env_context, shared_context)
            for fn in EXTENSION_FNS
        ],
    )


@jax.jit
def apply_extensions(grid, actions, per_env_context, shared_context, enable_extensions):
    """Apply extensions with their specified transformation skips using JAX control flow.

    Args:
        grid: Environment grid
        actions: Action array including extension choices
        per_env_context: Per-environment context
        shared_context: Shared context parameters
        enable_extensions: Whether extensions are enabled

    Returns:
        Array of extension channels
    """
    # Pre-compute all transformed grids
    transformed_grids = jnp.stack(
        [
            transform_grid(
                grid,
                per_env_context,
                skip_visibility=ext_info.skip_visibility,
                skip_blur=ext_info.skip_blur,
            )
            for reg in EXTENSION_REGISTRY
            for ext_info in reg.extensions
        ]
    )

    # Pre-compute extension indices
    extension_indices = jnp.array(
        [ext_info.index for reg in EXTENSION_REGISTRY for ext_info in reg.extensions]
    )

    def apply_single_extension(transformed_grid, ext_idx, action_slice):
        # Apply the selected extension function
        result = apply_extension_fn(
            ext_idx, transformed_grid, per_env_context, shared_context
        )
        return jnp.where(enable_extensions & action_slice, result, jnp.zeros_like(grid))

    # Apply all extensions in parallel using vmap
    extension_channels = jax.vmap(
        apply_single_extension,
        in_axes=(0, 0, 0),  # Map over transformed_grids and extension_indices
    )(transformed_grids, extension_indices, actions[..., 2:])

    return extension_channels


# Pre-generate extension functions at module level
def _generate_extension_functions():
    """Generate extension functions based on registry configuration."""
    extension_fns = []
    for reg in EXTENSION_REGISTRY:
        for ext_info in sorted(reg.extensions, key=lambda x: x.index):

            def make_fn(info):
                @jax.jit
                def fn(transformed_grid, per_env_context, shared_context):
                    return info.fn(
                        transformed_grid,
                        per_env_context,
                        shared_context,
                        skip_visibility=info.skip_visibility,
                        skip_blur=info.skip_blur,
                    )

                return fn

            extension_fns.append(make_fn(ext_info))
    return extension_fns


@flax.struct.dataclass
class ExtensionRegistry:
    """Registry for managing environment extensions."""

    extensions: tuple  # Tuple of ExtensionInfo objects
    choose: int  # Max number of extensions that can be active simultaneously


@flax.struct.dataclass
class ExtensionInfo:
    """Information about a single extension."""

    index: int
    fn: Callable
    skip_visibility: int = 0  # 0: apply visibility, 1: skip visibility
    skip_blur: int = 0  # 0: apply blur, 1: skip blur


# Extension registry configuration
EXTENSION_REGISTRY = [
    ExtensionRegistry(
        extensions=tuple(
            sorted(
                [
                    ExtensionInfo(
                        0, unblur_fn, skip_visibility=0, skip_blur=1
                    ),  # Skip blur but keep visibility
                    ExtensionInfo(
                        1, see_invisible_fires_fn, skip_visibility=1, skip_blur=0
                    ),  # Skip visibility but keep blur
                ],
                key=lambda x: x.index,
            )
        ),
        choose=1,  # Allow only one extension to be active at a time
    )
]

# Generate extension functions
EXTENSION_FNS = _generate_extension_functions()
