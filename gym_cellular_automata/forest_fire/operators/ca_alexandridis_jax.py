import jax.numpy as jnp
from jax import jit, vmap, random, lax
from gymnasium import spaces
from functools import partial
from gym_cellular_automata._config import TYPE_BOX
import numpy as np

from gym_cellular_automata.operator import Operator

import os


@partial(jit, static_argnums=(0))
def moore_n(n: int, position: tuple, grid: jnp.ndarray, invariant: float = 0.0):
    """Gets the N Moore neighborhood at given position using JAX.

    Args:
        n: Neighborhood radius
        position: (row, col) tuple of center position
        grid: Input grid
        invariant: Padding value for boundaries
    """
    # Pad the grid with invariant values
    padded = jnp.pad(grid, pad_width=n, mode="constant", constant_values=invariant)

    # Calculate slice indices (offset by padding)
    row, col = position
    row, col = row + n, col + n  # Adjust for padding

    # Extract neighborhood
    size = 2 * n + 1

    neighborhood = lax.dynamic_slice(
        padded, (row - n, col - n), (size, size)  # start_indices  # slice_sizes
    )
    return neighborhood


class PartiallyObservableForestFireJax(Operator):
    grid_dependant = True
    action_dependant = False
    context_dependant = True

    deterministic = False

    def __init__(self, empty, tree, fire, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.empty = empty
        self.tree = tree
        self.fire = fire

        if self.context_space is None:
            self.context_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=TYPE_BOX)

    # ... keep existing class setup ...

    @partial(jit, static_argnums=(0,))
    def _compute_burn_probability(self, vegetation, density, wind, slope):
        """Vectorized computation of burn probabilities"""
        # Convert discrete values to probabilities using JAX's type-safe operations
        p_veg = jnp.where(
            vegetation == 1,
            -0.3,
            jnp.where(
                vegetation == 2,
                0.0,
                jnp.where(vegetation == 3, 0.3, jnp.where(vegetation == 4, 0.6, 1.0)),
            ),
        )

        p_den = jnp.where(
            density == 1,
            -0.4,
            jnp.where(
                density == 2,
                0.0,
                jnp.where(density == 3, 0.3, jnp.where(density == 4, 0.6, 1.0)),
            ),
        )

        p_h = 0.58
        a = 0.078
        p_slope = jnp.exp(a * slope)

        p_veg = p_veg[..., None, None]  # Add two dimensions: (200, 200, 1, 1)
        p_den = p_den[..., None, None]  # Add two dimensions: (200, 200, 1, 1)

        return p_h * (1 + p_veg) * (1 + p_den) * wind * p_slope

    @partial(jit, static_argnums=(0,))
    def _compute_pinecone_burn_probability(self, vegetation, density):
        """Vectorized computation of pinecone burn probabilities"""
        p_veg = jnp.where(
            vegetation == 1,
            0.0,
            jnp.where(
                vegetation == 2,
                0.8,
                jnp.where(vegetation == 3, 1.6, jnp.where(vegetation == 4, 2.0, 2.5)),
            ),
        )

        p_den = jnp.where(
            density == 1,
            0.0,
            jnp.where(
                density == 2,
                0.6,
                jnp.where(density == 3, 1.2, jnp.where(density == 4, 1.5, 2.0)),
            ),
        )

        return 0.58 * (1 + p_veg) * (1 + p_den)

    @partial(jit, static_argnums=(0,))
    def _handle_pinecone_spread(self, grid, key, context, fire_mask, max_pinecones=5):
        """Handle pinecone dispersal and burning from fire cells with a fixed maximum number of pinecones.

        Args:
            grid: Current grid state
            key: JAX random key
            context: Simulation context
            fire_mask: Boolean mask of fire cells
            max_pinecones: Maximum number of pinecones per cell
        """
        ft = context["current_ft"]
        grid_height, grid_width = grid.shape

        # Sample number of pinecones for each cell (clipped to max)
        key1, key2, key3 = random.split(key, 3)
        n_pinecones = jnp.minimum(
            random.poisson(key1, 1.0, shape=grid.shape), max_pinecones
        )

        # Create arrays for all possible pinecones
        # Shape: (grid_height, grid_width, max_pinecones)
        directions = random.randint(
            key2, shape=(grid_height, grid_width, max_pinecones), minval=0, maxval=8
        )
        thrust = 3.0 * random.normal(
            key3, shape=(grid_height, grid_width, max_pinecones)
        )

        # Direction lookup tables
        dx = jnp.array([1, 1, 0, -1, -1, -1, 0, 1])  # E, NE, N, NW, W, SW, S, SE
        dy = jnp.array([0, 1, 1, 1, 0, -1, -1, -1])  # E, NE, N, NW, W, SW, S, SE
        ft_lookup = jnp.array(
            [
                (0, 0),
                (0, 1),
                (0, 2),  # Top row
                (1, 0),
                (1, 2),  # Middle row (skip center)
                (2, 0),
                (2, 1),
                (2, 2),  # Bottom row
            ]
        )
        row_indices = ft_lookup[directions][..., 0]
        col_indices = ft_lookup[directions][..., 1]

        # Adjust thrust by wind factor
        # thrust = thrust * ft[ft_indices]
        thrust = thrust * ft[row_indices, col_indices]

        # Create source position arrays
        rows = jnp.arange(grid_height)[:, None, None]  # Shape: (height, 1, 1)
        cols = jnp.arange(grid_width)[None, :, None]  # Shape: (1, width, 1)

        # Calculate landing positions for all potential pinecones
        new_rows = jnp.clip(
            jnp.round(rows + dx[directions] * thrust), 0, grid_height - 1
        ).astype(jnp.int32)

        new_cols = jnp.clip(
            jnp.round(cols + dy[directions] * thrust), 0, grid_width - 1
        ).astype(jnp.int32)

        # Create validity mask for each pinecone
        pinecone_mask = fire_mask[:, :, None] & (  # Only from fire cells
            jnp.arange(max_pinecones)[None, None, :] < n_pinecones[:, :, None]
        )  # Only valid pinecones

        # Compute burn probabilities
        pinecone_probs = self._compute_pinecone_burn_probability(
            context["vegetation"], context["density"]
        )

        key, subkey = random.split(key)
        pinecone_random = random.uniform(
            subkey, shape=(grid_height, grid_width, max_pinecones)
        )

        # Create masks for valid landings and burns
        landing_mask = (grid[new_rows, new_cols] == self.tree) & pinecone_mask
        burn_mask = landing_mask & (
            pinecone_random < pinecone_probs[new_rows, new_cols]
        )

        # Flatten arrays for grid updates
        new_rows = new_rows.reshape(-1)
        new_cols = new_cols.reshape(-1)
        burn_mask = burn_mask.reshape(-1)

        return new_rows, new_cols, burn_mask

    @partial(jit, static_argnums=(0,))
    def _update_grid(self, grid, key, context):
        """Single step of grid update using JAX operations"""
        wind_matrix = context["current_wind_matrix"]
        # Create masks for different cell states
        tree_mask = grid == self.tree
        fire_mask = grid == self.fire
        empty_mask = grid == self.empty

        # Get neighborhoods for all cells

        rows, cols = jnp.meshgrid(
            jnp.arange(grid.shape[0]), jnp.arange(grid.shape[1]), indexing="ij"
        )
        neighborhoods = vmap(
            vmap(moore_n, in_axes=(None, 0, None)), in_axes=(None, 0, None)
        )(1, (rows, cols), grid)

        # Handle direct fire spread
        key, subkey = random.split(key)
        burn_probs = self._compute_burn_probability(
            context["vegetation"],
            context["density"],
            wind_matrix,
            context["slope"],
        )

        random_values_burn = random.uniform(key, burn_probs.shape)
        key, subkey = random.split(key)
        random_values_grow = random.uniform(key, grid.shape)

        # Sets to fire if the following:
        # - If the cell is a tree
        # - If any of the neighbors are on fire
        # - If the any of the on fire neighbor passes a burn prob based on the
        #   neighbors wind and slope and the current cells vegetation and density
        # Then does else for growing trees and empty cells

        new_grid = jnp.where(
            tree_mask
            & ((neighborhoods == self.fire) & (random_values_burn < burn_probs)).any(
                axis=(-1, -2)
            ),
            self.fire,
            jnp.where(
                empty_mask & (random_values_grow < context["p_tree"]),
                self.tree,
                jnp.where(fire_mask & (context["fire_age"] <= 1), self.empty, grid),
            ),
        )

        # Handle pinecone dispersal
        key, subkey = random.split(key)
        new_rows, new_cols, burn_mask = self._handle_pinecone_spread(
            new_grid, subkey, context, fire_mask
        )

        # Update grid with pinecone effects
        new_grid = new_grid.at[new_rows, new_cols].set(
            jnp.where(burn_mask, self.fire, new_grid[new_rows, new_cols])
        )

        # Update fire age
        new_fire_age = jnp.where(
            fire_mask, context["fire_age"] - 1, context["fire_age"]
        )
        return new_grid, new_fire_age

    def update(self, grid, action, context):
        wind_matrix, ft = context["winds"][context["wind_index"]]

        # Create new context with selected wind matrix
        jitted_context = {
            **context,
            "current_wind_matrix": wind_matrix,
            "current_ft": ft,
        }
        key = random.PRNGKey(self.np_random.integers(0, 2**32))
        new_grid, new_fire_age = self._update_grid(grid, key, jitted_context)

        # Handle wind changes
        key, subkey = random.split(key)
        wind_change = random.uniform(subkey) < context["p_wind_change"]
        new_wind_index = jnp.where(
            wind_change,
            (context["wind_index"] + random.randint(subkey, (), 1, 8))
            % len(context["winds"]),
            context["wind_index"],
        )

        context = dict(context)
        context["fire_age"] = new_fire_age
        context["wind_index"] = new_wind_index
        # Convert JAX array back to numpy before returning
        # new_grid = np.array(new_grid)

        return new_grid, context
