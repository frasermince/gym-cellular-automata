import matplotlib
import pytest
import jax.numpy as jnp
import jax

from gym_cellular_automata.forest_fire.bulldozer.advanced_bulldozer import (
    AdvancedForestFireBulldozerEnv,
)
from gym_cellular_automata.grid_space import GridSpace

THRESHOLD = 12

NROWS, NCOLS = 256, 256


@pytest.fixture
def env():
    env = AdvancedForestFireBulldozerEnv(
        nrows=NROWS,
        ncols=NCOLS,
        debug=True,
        enable_extensions=True,
        key=jax.random.key(0),
    )
    env.reset()
    return env


def test_extension_observations(env):
    """Test that different extension actions produce correct observation channels"""
    obs, info = env.reset()
    grid, context = obs

    # Create test actions with different extension choices
    # Base action (move, shoot) + extension choice
    actions = jnp.array(
        [
            [4, 0, 0],  # No extensions active
            [4, 0, 1],  # First extension active (unblur)
            [4, 0, 2],  # Second extension active (see invisible fires)
            [4, 0, 3],  # Both extensions active
        ]
    )

    # Step environment with each action
    for i, action in enumerate(actions):
        action_batch = jnp.tile(action[None], (env.num_envs, 1))
        next_obs, _, _, _, next_info = env.stateless_step(action_batch, obs, info)
        next_grid, next_context = next_obs

        # Update obs and info for next iteration
        obs = next_obs
        info = next_info

        # Check observation channels for each action type
        if i == 0:  # No extensions
            # Extension channels should be all zeros
            assert not jnp.all(next_grid[..., 0] == 0), "Base grid should not be zero"
            assert jnp.any(
                next_grid[..., 1] == 1
            ), "Positional channel should contain a 1"
            assert jnp.all(
                next_grid[..., 2:] == 0
            ), "Extension channels should be zero when no extensions active"

        if i == 1 or i == 3:  # Unblur extension
            if i == 1:
                assert jnp.all(
                    next_grid[..., 3] == 0
                ), "Invisible fires channel should be zero"

            # Extension channel should show unblurred view
            unblur_channel = next_grid[..., 2]
            has_fires = jnp.any(next_grid[..., 0] == env._fire, axis=(1, 2))

            # Check if grids differ where they should
            grids_equal = jnp.all(unblur_channel == next_grid[..., 0], axis=(1, 2))

            # Assert that grids differ when they should
            assert not jnp.any(
                jnp.logical_and(has_fires, grids_equal)
            ), "Unblur channel should differ from base grid during day with fires"

        if i == 2 or i == 3:  # See invisible fires extension

            if i == 2:
                assert jnp.all(next_grid[..., 2] == 0), "Unblur channel should be zero"
            # Extension channel should show fires even during day
            invisible_fires_channel = next_grid[..., 3]
            is_daytime = next_context["per_env_context"]["is_night"] == 0

            # Only check daytime environments
            daytime_envs = jnp.where(is_daytime)[0]
            if len(daytime_envs) > 0:
                # Count fires in base grid and extension channel
                base_fires = jnp.sum(
                    next_grid[daytime_envs, ..., 0] == env._fire, axis=(1, 2)
                )
                channel_fires = jnp.sum(
                    invisible_fires_channel[daytime_envs] == env._fire, axis=(1, 2)
                )

                assert jnp.all(base_fires == 0), "Base grid should have no fires"
                assert jnp.all(
                    channel_fires >= base_fires
                ), "Invisible fires channel should show at least as many fires as base grid"


def test_extension_action_space(env):
    """Test that extension action space is correctly configured"""
    # Check total action space includes extensions
    assert (
        env.total_action_space.shape[-1] == 3
    ), "Action space should have 3 dimensions (move, shoot, extensions)"

    # Check extension choices match registry
    assert len(env.extension_choices) == 1, "Should have 1 extension registry"
    assert env.extension_choices[0] == (
        2,
        1,
    ), "Registry should allow choosing 1 from 2 extensions"

    # Verify extension lookups are correct shape
    assert len(env._extension_lookups) == 1, "Should have 1 extension lookup table"
    assert (
        env._extension_lookups[0].shape[1] == 2
    ), "Lookup table should map to 2 binary choices"
