from typing import Optional, Tuple

import jax.numpy as jnp
import jax
import numpy as np
from jax import jit, vmap, random, lax
from gymnasium import spaces

from gym_cellular_automata._config import TYPE_BOX, TYPE_INT
from gym_cellular_automata.ca_env import CAEnv
from gym_cellular_automata.forest_fire.operators import (
    ModifyJax,
    MoveJax,
    MoveModifyJax,
    RepeatCAJax,
    PartiallyObservableForestFire,
    PartiallyObservableForestFireJax,
)
from gym_cellular_automata.grid_space import GridSpace
from gym_cellular_automata.operator import Operator

from .utils.advanced_bulldozer_render import render, plot_grid_attribute

import math
from functools import partial


def init_vegetation(row_count, column_count, num_envs):
    veg_matrix = np.zeros((num_envs, row_count, column_count), dtype=int)

    for env in range(num_envs):
        # Create random patches of different vegetation types
        num_patches = np.random.randint(4, 8)
        for _ in range(num_patches):
            # Random patch center and size
            center_row = np.random.randint(0, row_count)
            center_col = np.random.randint(0, column_count)
            patch_height = np.random.randint(3, row_count // 2)
            patch_width = np.random.randint(3, column_count // 2)

            # Random vegetation type (1-3)
            veg_type = np.random.randint(1, 6)

            # Fill patch
            row_start = max(0, center_row - patch_height // 2)
            row_end = min(row_count, center_row + patch_height // 2)
            col_start = max(0, center_col - patch_width // 2)
            col_end = min(column_count, center_col + patch_width // 2)

            veg_matrix[env, row_start:row_end, col_start:col_end] = veg_type

        # Fill any remaining zeros randomly
        zero_mask = veg_matrix[env] == 0
        veg_matrix[env][zero_mask] = np.random.randint(1, 4, size=np.sum(zero_mask))

    return veg_matrix


def init_density(row_count, column_count, num_envs):
    den_matrix = np.zeros((num_envs, row_count, column_count), dtype=int)

    for env in range(num_envs):
        # Create random patches of different density types
        num_patches = np.random.randint(4, 8)
        for _ in range(num_patches):
            # Random patch center and size
            center_row = np.random.randint(0, row_count)
            center_col = np.random.randint(0, column_count)
            patch_height = np.random.randint(3, row_count // 2)
            patch_width = np.random.randint(3, column_count // 2)

            # Random density type (1-3)
            den_type = np.random.randint(1, 6)

            # Fill patch
            row_start = max(0, center_row - patch_height // 2)
            row_end = min(row_count, center_row + patch_height // 2)
            col_start = max(0, center_col - patch_width // 2)
            col_end = min(column_count, center_col + patch_width // 2)

            den_matrix[env, row_start:row_end, col_start:col_end] = den_type

        # Fill any remaining zeros randomly
        zero_mask = den_matrix[env] == 0
        den_matrix[env][zero_mask] = np.random.randint(1, 4, size=np.sum(zero_mask))

    return den_matrix


def init_density_same(row_count, column_count, num_envs):
    """Creates a density matrix filled entirely with middle value (3)"""
    den_matrix = np.full((num_envs, row_count, column_count), 3, dtype=int)
    return den_matrix


def init_vegetation_same(row_count, column_count, num_envs):
    """Creates a vegetation matrix filled entirely with middle value (3)"""
    veg_matrix = np.full((num_envs, row_count, column_count), 3, dtype=int)
    return veg_matrix


def init_altitude_same(row_count, column_count, num_envs):
    altitude = np.full((num_envs, row_count, column_count), 0, dtype=int)
    return altitude


def init_altitude(row_count, column_count, num_envs):
    altitude = np.zeros((num_envs, row_count, column_count))

    for env in range(num_envs):
        # Start with a base of random noise
        altitude[env] = np.random.uniform(0, 5, (row_count, column_count))

        # Add more small hills
        num_hills = np.random.randint(6, 10)  # More hills
        for _ in range(num_hills):
            # Random hill center and size
            center_row = np.random.randint(0, row_count)
            center_col = np.random.randint(0, column_count)
            radius = np.random.randint(
                2, min(row_count, column_count) // 4
            )  # Smaller radius
            height = np.random.uniform(2, 6)  # Lower heights for smaller hills

            # Create hill using distance from center
            for i in range(row_count):
                for j in range(column_count):
                    distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
                    if distance < radius:
                        # Smooth hill falloff using cosine
                        factor = np.cos(distance / radius * np.pi / 2)
                        altitude[env, i, j] += height * factor

        # Add some gentle slopes between flat areas
        num_slopes = np.random.randint(4, 8)  # More slopes
        for _ in range(num_slopes):
            # Random rectangular area - smaller areas
            start_row = np.random.randint(0, row_count - 4)
            start_col = np.random.randint(0, column_count - 4)
            width = np.random.randint(3, column_count // 4)  # Smaller width
            height = np.random.randint(3, row_count // 4)  # Smaller height

            # Small height difference for gentle slope
            height_diff = np.random.uniform(1, 4)  # Smaller height differences

            # Add gradual slope
            for i in range(start_row, min(start_row + height, row_count)):
                for j in range(start_col, min(start_col + width, column_count)):
                    progress = (i - start_row) / height
                    altitude[env, i, j] += height_diff * progress

    return altitude / 10


def tg(x):
    return math.degrees(math.atan(x))


def get_slope(altitude, row_count, column_count, num_envs):
    # Initialize slope matrix with 3x3 sub-matrices of zeros
    slope_matrix = np.zeros((num_envs, row_count, column_count, 3, 3))

    for env in range(num_envs):
        # Skip edges since they remain flat (all zeros)
        for row in range(1, row_count - 1):
            for col in range(1, column_count - 1):
                # Get 3x3 neighborhood of altitudes
                neighborhood = altitude[env, row - 1 : row + 2, col - 1 : col + 2]
                current = altitude[env, row, col]

                # Calculate slopes
                diffs = current - neighborhood

                # Apply diagonal scaling
                diagonals = [(0, 0), (0, 2), (2, 0), (2, 2)]
                for i, j in diagonals:
                    diffs[i, j] /= 1.414

                # Convert to angles
                slope_matrix[env, row, col] = np.degrees(np.arctan(diffs))

                # Center point is always 0
                slope_matrix[env, row, col, 1, 1] = 0

    # Bin slope values into 10 ranges and count cells in each bin
    slope_flat = slope_matrix.flatten()
    slope_min, slope_max = np.min(slope_flat), np.max(slope_flat)
    bins = np.linspace(slope_min, slope_max, num=11)  # 11 edges make 10 bins
    hist, _ = np.histogram(slope_flat, bins=bins)
    print("\nSlope distribution:")
    for i, count in enumerate(hist):
        print(f"Range {bins[i]:.2f} to {bins[i+1]:.2f}: {count} cells")
    return slope_matrix


wind_thetas = [
    # North Wind (blowing from South to North)
    [[45, 0, 45], [90, 0, 90], [135, 180, 135]],
    # Northeast Wind
    [[90, 45, 0], [135, 0, 45], [180, 135, 90]],
    # East Wind (blowing from West to East)
    [[135, 90, 45], [180, 0, 0], [135, 90, 45]],
    # Southeast Wind
    [[180, 135, 90], [135, 0, 45], [90, 45, 0]],
    # South Wind (blowing from North to South)
    [[135, 180, 135], [90, 0, 90], [45, 0, 45]],
    # Southwest Wind
    [[90, 135, 180], [45, 0, 135], [0, 45, 90]],
    # West Wind (blowing from East to West)
    [[45, 90, 135], [0, 0, 180], [45, 90, 135]],
    # Northwest Wind
    [[0, 45, 90], [45, 0, 135], [90, 135, 180]],
]

# thetas = [[45, 0, 45], [90, 0, 90], [135, 180, 135]]


def calc_pw(theta):
    c_1, c_2 = 0.045, 0.131
    V = 10
    t = np.radians(theta)
    ft = np.exp(V * c_2 * (np.cos(t) - 1))
    return np.exp(c_1 * V) * ft, ft


def get_winds(use_hidden):
    winds = []
    if use_hidden:
        thetas = wind_thetas
    else:
        thetas = [wind_thetas[0] for _ in range(len(wind_thetas))]
    for thetas in wind_thetas:
        wind_matrix = np.zeros((3, 3))
        theta_array = np.array(thetas)
        wind_matrix, ft = calc_pw(theta_array)
        wind_matrix[1, 1] = 0
        winds.append((wind_matrix, ft))
    return winds


class AdvancedForestFireBulldozerEnv(CAEnv):
    metadata = {"render_modes": ["human"], "render_mode": "rgb_array"}

    @property
    def MDP(self):
        return self._MDP

    @property
    def initial_state(self):
        if self._resample_initial:
            self.grid = self._initial_grid_distribution()
            self.context = self._initial_context_distribution()

            self._initial_state = self.grid, self.context

        self._resample_initial = False

        return self._initial_state

    def __init__(
        self,
        nrows,
        ncols,
        key,
        num_envs=8,
        speed_move=0.12,
        speed_act=0.03,
        pos_bull: Optional[Tuple] = None,
        pos_fire: Optional[Tuple] = None,
        t_move: Optional[float] = None,
        t_shoot: Optional[float] = None,
        t_any=0.001,
        p_tree=0.90,
        p_empty=0.10,
        use_hidden: bool = True,
        middle_fire: bool = False,
        **kwargs,
    ):
        super().__init__(nrows, ncols, **kwargs)
        self.middle_fire = middle_fire
        self.use_hidden = use_hidden
        self.starting_key = key

        self.shared_context_keys = {
            "winds",  # Shared wind patterns
            "p_fire",  # Global probability parameters
            "p_tree",
            "p_wind_change",
        }

        self.per_env_context_keys = {
            "wind_index",
            "density",
            "vegetation",
            "altitude",
            "slope",
            "fire_age",
            "key",
        }

        self.num_envs = num_envs
        self.title = "ForestFireBulldozer" + str(nrows) + "x" + str(ncols)

        # Env Representation Parameters

        (
            up_left,
            up,
            up_right,
            left,
            not_move,
            right,
            down_left,
            down,
            down_right,
        ) = range(
            9
        )  # Move actions

        self._shoots = {"shoot": 1, "none": 0}  # Shoot actions

        self._empty = 0  # Empty cell
        self._tree = 3  # Tree cell
        self._fire = 25  # Fire cell

        # Initial Condition Parameters

        self._pos_bull = (
            pos_bull  # Initial position of bulldozer, default at `initial_context`
        )
        self._pos_fire = (
            [pos_fire] * num_envs if pos_fire is not None else None
        )  # Initial position of fire for each env, default at `initial_fire`

        self._p_tree_init = p_tree  # Initial Tree probability
        self._p_empty_init = p_empty  # Initial Empty probality

        # Env Behavior Parameters

        winds = get_winds(use_hidden)  # Get numpy array

        if use_hidden:
            density = init_density(nrows, ncols, num_envs)

            vegetation = init_vegetation(nrows, ncols, num_envs)

            altitude = init_altitude(nrows, ncols, num_envs)
        else:
            density = init_density_same(nrows, ncols, num_envs)

            vegetation = init_vegetation_same(nrows, ncols, num_envs)

            altitude = init_altitude_same(nrows, ncols, num_envs)

        slope = get_slope(altitude, nrows, ncols, num_envs)

        self._winds = jnp.array(winds)  # Convert to JAX array
        self._wind = self._winds[0]
        self._density = jnp.array(density)  # Convert to JAX array
        self._vegitation = jnp.array(vegetation)  # Convert to JAX array
        self._altitude = jnp.array(altitude)  # Convert to JAX array
        self._slope = jnp.array(slope)  # Convert to JAX array

        self._fire_age = jnp.zeros((num_envs, nrows, ncols))
        self._p_fire = 0.00033
        self._p_tree = 0.00333
        self._p_wind_change = 0.06

        self._effects = {self._tree: self._empty}  # Substitution Effect

        # Time to do things
        # On Cellular Automaton (CA) updates units
        # e.g. 1 equals to 1 CA update

        self._t_env_any = t_any  # Time regardless of anything, guarantees termination
        self._t_act_none = 0.0  # Time of doing NOTHING

        # Scale dependent times
        # They depend on grid scale and speed

        # Two speed params are defined

        # speed_move:
        # percent of the grid that the bulldozer can cover by only moving
        # before an update happens

        # speed_act:
        # percent of the grid that the bulldozer can cover while acting and moving
        # before an update happens

        scale = (self.nrows + self.ncols) // 2
        # Time of moving
        self._t_act_move = (
            (1 / (speed_move * scale)) - t_any if t_move is None else t_move
        )
        # Time of shooting
        self._t_act_shoot = (
            (1 / (speed_act * scale)) - self._t_act_move if t_shoot is None else t_shoot
        )

        # For `_init_time_mappings`
        self._moves = {
            "up_left": up_left,
            "up": up,
            "up_right": up_right,
            "left": left,
            "not_move": not_move,
            "right": right,
            "down_left": down_left,
            "down": down,
            "down_right": down_right,
        }

        # For `MoveModify`
        self._action_sets = {
            "up": {up_left, up, up_right},
            "down": {down_left, down, down_right},
            "left": {up_left, left, down_left},
            "right": {up_right, right, down_right},
            "not_move": {not_move},
        }

        self._set_spaces()
        self._init_time_mappings()

        # self.ca = PartiallyObservableForestFireJax(
        #     self._empty, self._tree, self._fire, **self.ca_space
        # )
        self.ca = PartiallyObservableForestFireJax(
            self._empty, self._tree, self._fire, **self.ca_space
        )

        self.move = MoveJax(self._action_sets, **self.move_space)
        self.modify = ModifyJax(self._effects, **self.modify_space)

        # Composite Operators
        self.move_modify = MoveModifyJax(
            self.move, self.modify, **self.move_modify_space
        )
        self.repeater = RepeatCAJax(
            self.ca, self.time_per_action, self.time_per_state, **self.repeater_space
        )
        self._MDP = MDP(self.repeater, self.move_modify, **self.MDP_space)

    # Gym API
    # step, reset & seed methods inherited from parent class

    @partial(jax.jit, static_argnums=0)
    def stateless_step(self, action, obs, info):
        # MDP Transition
        grid, context = obs

        shared_context = context["shared_context"]
        per_env_context = context["per_env_context"]
        per_env_in_axes = {k: 0 for k in self.per_env_context_keys}
        next_grid, updated_context = jax.vmap(
            self.MDP,
            in_axes=(
                0,
                0,
                per_env_in_axes,
                None,
                0,
                0,
            ),  # grid, action, per_env_context, shared_context
        )(
            grid,
            action,
            per_env_context,
            shared_context,
            context["position"],
            context["time"],
        )
        per_env_portion, next_position, next_time = updated_context
        context["per_env_context"] = per_env_portion
        context["position"] = next_position
        context["time"] = next_time

        next_state = (next_grid, context)

        # Check for termination
        next_done = jax.vmap(self._is_done, in_axes=0)(next_grid)

        # Gym API Formatting
        obs = next_state
        reward = jax.vmap(self._award, in_axes=(0, 0))(grid, next_grid)
        terminated = next_done
        truncated = jnp.full(grid.shape[0], False)
        info["reward"] = reward
        info["terminated"] = terminated
        info["TimeLimit.truncated"] = truncated

        info["steps_elapsed"] = info["steps_elapsed"] + jnp.ones(1)
        info["reward_accumulated"] = info["reward_accumulated"] + jnp.array(reward)

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    @partial(jax.jit, static_argnums=(0,), static_argnames=("seed", "options"))
    def conditional_reset(
        self, step_tuple, *, seed: Optional[int] = None, options: Optional[dict] = None
    ):
        def reset_fn(args):
            step_tuple, (initial_grid, initial_context) = args
            obs, reward, terminated, truncated, info = step_tuple
            grid, context = obs
            grid = jnp.where(terminated[:, None, None], initial_grid, grid)

            for key in self.per_env_context_keys:
                key_shape = context["per_env_context"][key].shape
                # Add as many dimensions as needed after the batch dimension
                expanded_terminated = terminated[
                    (...,) + (None,) * (len(key_shape) - 1)
                ]
                context["per_env_context"][key] = jnp.where(
                    expanded_terminated,  # adjust broadcasting as needed
                    initial_context["per_env_context"][key],
                    context["per_env_context"][key],
                )
            obs = (grid, context)
            info["steps_elapsed"] = jnp.where(terminated, 0, info["steps_elapsed"])
            info["reward_accumulated"] = jnp.where(
                terminated, 0.0, info["reward_accumulated"]
            )

            new_terminated = jnp.zeros_like(terminated, dtype=bool)

            return obs, reward, new_terminated, truncated, info

        return jax.lax.cond(
            step_tuple[2].any(),
            reset_fn,
            lambda x: x[0],
            (step_tuple, self.initial_state),
        )

    def render(self, mode="human"):
        return render(self)

    def altitude_render(self):
        altitudes = []
        for a in range(self._altitude.shape[0]):
            altitudes.append(plot_grid_attribute(self._altitude[a], "Altitude"))
        return altitudes

    def density_render(self):
        densities = []
        for d in range(self._density.shape[0]):
            densities.append(plot_grid_attribute(self._density[d], "Density"))
        return densities

    def vegitation_render(self):
        vegitations = []
        for v in range(self._vegitation.shape[0]):
            vegitations.append(plot_grid_attribute(self._vegitation[v], "Vegitation"))
        return vegitations

    # def _award(self, grid):
    #     #     """Reward Function

    #     #     Negative Ratio of Burning Area per Total Flammable Area

    #     #     -(f / (t + f))
    #     #     Where:
    #     #         t: tree cell counts
    #     #         f: fire cell counts

    #     #     Objective:
    #     #     Keep as much forest as possible.

    #     #     Advantages:
    #     #     1. Easy to interpret.
    #     #         + Percent of the forest lost at each step.
    #     #     2. Terminate ASAP.
    #     #         + As the reward is negative.
    #     #     3. Built-in cost of action.
    #     #         + The agent removes trees, this decreases the reward.
    #     #     4. Shaped reward.
    #     #         + Reward is given at each step.

    #     #     Disadvantages:
    #     #     1. Lack of experimental results.
    #     #     2. Is it equivalent with Sparse Reward?

    #     #     The sparse reward is alive trees at epidose's end:
    #     #     t / (e + t + f)
    #     #     """
    #     counts = self.count_cells(grid)

    #     t = counts[self._tree]
    #     f = counts[self._fire]
    #     return -(f / (t + f))
    def _award(self, prev_grid, grid):
        total_cells = float(self.nrows * self.ncols)

        # Current state reward (bounded [-1, 1])
        counts = self.count_cells(grid)
        state_reward = (
            -0.2 * counts[self._empty]
            + 1.0 * counts[self._tree]
            + -1.0 * counts[self._fire]
        ) / total_cells

        # Change-based rewards (each change is bounded [-1, 1])
        prev_counts = self.count_cells(prev_grid)
        fire_change = (counts[self._fire] - prev_counts[self._fire]) / total_cells
        tree_change = (counts[self._tree] - prev_counts[self._tree]) / total_cells
        empty_change = (counts[self._empty] - prev_counts[self._empty]) / total_cells

        # Scale factors for combining rewards
        STATE_WEIGHT = 1.0
        CHANGE_WEIGHT = 0.5

        reward = (
            STATE_WEIGHT * state_reward  # [-1, 1]
            + CHANGE_WEIGHT
            * (
                (-1.0 * fire_change)  # [-0.5, 0.5]
                + (0.25 * tree_change)  # [-0.125, 0.125]
                + (-0.05 * empty_change)  # [-0.025, 0.025]
            )
            + -0.01  # Time pressure
        )

        return reward

    # def _award(self, prev_grid, grid):
    #     prev_counts = self.count_cells(prev_grid)
    #     counts = self.count_cells(grid)
    #     t = counts[self._tree]  # trees
    #     f = counts[self._fire]  # fires
    #     e = counts[self._empty]  # empty
    #     total_cells = t + f + e

    #     # Reward for preventing tree loss
    #     tree_change = (counts[self._tree] - prev_counts[self._tree]) / total_cells
    #     fire_change = (counts[self._fire] - prev_counts[self._fire]) / total_cells
    #     return tree_change * 5.0 + -fire_change * 10.0

    def _is_done(self, grid):
        return jnp.invert(jnp.any(grid == self._fire))

    def _report(self):
        return {"hit": self.modify.hit}

    def _noise(self, ax_len):
        """
        Noise to initial conditions.
        """
        AX_PERCENT = 1 / 12
        upper = int(ax_len * AX_PERCENT)

        if upper > 0:
            return self.np_random.choice(range(upper), size=1).item(0)
        else:
            return 0

    def _initial_grid_distribution(self):
        # fmt: off
        grid_space = GridSpace(
            values = [  self._empty,   self._tree,   self._fire],
            probs  = [self._p_empty_init, self._p_tree_init,          0.0],
            shape=(self.num_envs, self.nrows, self.ncols),
        )
        # fmt: on

        grid = jnp.array(grid_space.sample())

        # Fire Position
        # Around the lower left quadrant for each env
        if self._pos_fire is None:
            if self.middle_fire:
                self._pos_fire = []
                for _ in range(self.num_envs):
                    r = (self.nrows // 2) + self._noise(self.nrows)
                    c = (self.ncols // 2) + self._noise(self.ncols)
                    self._pos_fire.append((r, c))

            else:
                self._pos_fire = []
                for _ in range(self.num_envs):
                    r = (3 * self.nrows // 4) + self._noise(self.nrows)
                    c = (1 * self.ncols // 4) + self._noise(self.ncols)
                    self._pos_fire.append((r, c))

        for env in range(self.num_envs):
            r, c = self._pos_fire[env]
            grid = grid.at[env, r, c].set(self._fire)
            self._fire_age = self._fire_age.at[env, r, c].set(10)

        return jnp.array(grid, dtype=jnp.int32)

    def _initial_context_distribution(self):
        init_time = jnp.zeros(self.num_envs)

        if self._pos_bull is None:
            # Bulldozer Position for each env
            # Around the upper right quadrant
            self._pos_bull = []
            for _ in range(self.num_envs):
                r, c = (1 * self.nrows // 4), (3 * self.ncols // 4)

                r = r + self._noise(self.nrows)
                c = c + self._noise(self.ncols)

                self._pos_bull.append((r, c))

        init_position = jnp.array(self._pos_bull)

        # Per-environment context parameters
        new_keys = jax.random.split(self.starting_key, self.num_envs)
        per_env_context = {
            "wind_index": jnp.array(
                (
                    self.np_random.integers(0, 8, size=self.num_envs)
                    if self.use_hidden
                    else jnp.zeros(self.num_envs)
                ),
                dtype=jnp.int32,
            ),
            "density": self._density,
            "vegetation": self._vegitation,
            "altitude": self._altitude,
            "slope": self._slope,
            "fire_age": self._fire_age,
            "key": new_keys,
        }

        # Shared context parameters
        shared_context = {
            "winds": self._winds,
            "p_fire": jnp.array(self._p_fire, dtype=jnp.float32),
            "p_tree": jnp.array(self._p_tree, dtype=jnp.float32),
            "p_wind_change": jnp.array(self._p_wind_change, dtype=jnp.float32),
        }

        init_context = {
            "per_env_context": per_env_context,
            "shared_context": shared_context,
            "position": init_position,
            "time": init_time,
        }

        return init_context

    def _init_time_mappings(self):
        self._movement_timings = {
            move: self._t_act_move for move in self._moves.values()
        }
        self._shooting_timings = {
            shoot: self._t_act_shoot for shoot in self._shoots.values()
        }

        self._movement_timings[self._moves["not_move"]] = self._t_act_none
        self._shooting_timings[self._shoots["none"]] = self._t_act_none
        self._jax_movement_timings = jnp.array(
            [self._movement_timings[k] for k in sorted(self._movement_timings.keys())]
        )
        self._jax_shooting_timings = jnp.array(
            [self._shooting_timings[k] for k in sorted(self._shooting_timings.keys())]
        )

        @jit
        def time_per_action(action, movement_timings, shooting_timings):

            time_on_move = movement_timings[action[0]]
            time_on_shoot = shooting_timings[action[1]]

            return time_on_move + time_on_shoot

        # Create a wrapper that provides the timings
        def time_per_action_wrapper(action):
            return time_per_action(
                action, self._jax_movement_timings, self._jax_shooting_timings
            )

        self.time_per_action = time_per_action_wrapper
        self.time_per_state = lambda s: jnp.array(self._t_env_any)

    def _parse_wind(self, windD: dict) -> np.ndarray:
        from gymnasium import spaces

        # fmt: off
        wind = np.array(
            [
                [ windD["up_left"]  , windD["up"]  , windD["up_right"]   ],
                [ windD["left"]     ,    0.0       , windD["right"]      ],
                [ windD["down_left"], windD["down"], windD["down_right"] ],
            ], dtype=TYPE_BOX
        )

        # fmt: on
        wind_space = spaces.Box(0.0, 1.0, shape=(3, 3), dtype=TYPE_BOX)

        assert wind_space.contains(wind), "Bad Wind Data, check ranges [0.0, 1.0]"

        return wind

    def _set_spaces(self):
        self.grid_space = GridSpace(
            values=[self._empty, self._tree, self._fire],
            shape=(self.num_envs, self.nrows, self.ncols),
        )

        # Create spaces for per-env context parameters
        self.per_env_context_space = {
            "wind_index": spaces.Box(
                0, len(self._winds) - 1, shape=(self.num_envs,), dtype=TYPE_INT
            ),
            "density": spaces.Box(
                0, 5, shape=(self.num_envs, self.nrows, self.ncols), dtype=TYPE_INT
            ),
            "vegetation": spaces.Box(
                0, 5, shape=(self.num_envs, self.nrows, self.ncols), dtype=TYPE_INT
            ),
            "altitude": spaces.Box(
                0.0,
                float("inf"),
                shape=(self.num_envs, self.nrows, self.ncols),
                dtype=TYPE_BOX,
            ),
            "slope": spaces.Box(
                -90.0,
                90.0,
                shape=(self.num_envs, self.nrows, self.ncols, 3, 3),
                dtype=TYPE_BOX,
            ),
            "fire_age": spaces.Box(
                0,
                float("inf"),
                shape=(self.num_envs, self.nrows, self.ncols),
                dtype=TYPE_BOX,
            ),
        }

        # Create spaces for shared context parameters
        self.shared_context_space = {
            "winds": spaces.Box(0.0, 1.0, shape=(8, 3, 3), dtype=TYPE_BOX),
            "p_fire": spaces.Box(0.0, 1.0, shape=(), dtype=TYPE_BOX),
            "p_tree": spaces.Box(0.0, 1.0, shape=(), dtype=TYPE_BOX),
            "p_wind_change": spaces.Box(0.0, 1.0, shape=(), dtype=TYPE_BOX),
        }

        self.position_space = spaces.Box(
            low=np.zeros((self.num_envs, 2), dtype=TYPE_INT),
            high=np.tile(np.array([self.nrows, self.ncols]), (self.num_envs, 1)),
            shape=(self.num_envs, 2),
            dtype=TYPE_INT,
        )
        self.time_space = spaces.Box(
            0.0, float("inf"), shape=(self.num_envs,), dtype=TYPE_BOX
        )

        self.context_space = spaces.Dict(
            {
                "per_env_context": spaces.Dict(self.per_env_context_space),
                "shared_context": spaces.Dict(self.shared_context_space),
                "position": self.position_space,
                "time": self.time_space,
            }
        )

        # RL spaces
        m, n = len(self._moves), len(self._shoots)
        self.action_space = spaces.MultiDiscrete(
            nvec=np.array([[m, n]] * self.num_envs),  # Shape becomes (num_envs, 2)
            dtype=TYPE_INT,
        )
        self.observation_space = spaces.Tuple((self.grid_space, self.context_space))

        # Suboperators Spaces
        self.ca_space = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.context_space,
        }

        self.move_space = {
            "grid_space": self.grid_space,
            "action_space": spaces.Discrete(m),
            "context_space": self.position_space,
        }

        self.modify_space = {
            "grid_space": self.grid_space,
            "action_space": spaces.Discrete(n),
            "context_space": self.position_space,
        }

        self.move_modify_space = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.position_space,
        }

        self.repeater_space = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.context_space,
        }

        self.MDP_space = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.context_space,
        }

    def count_cells(self, grid=None):
        """Returns dict of cell counts by using jax.numpy operations"""

        grid = self.grid if grid is None else grid
        # Assuming your grid contains values 0, 1, 2, etc.
        # Create a dict with counts for each possible value
        counts = {
            self._empty: jnp.sum(grid == self._empty),
            self._tree: jnp.sum(grid == self._tree),
            self._fire: jnp.sum(grid == self._fire),
        }

        return counts


class MDP(Operator):
    grid_dependant = True
    action_dependant = True
    context_dependant = True

    deterministic = False

    def __init__(self, repeat_ca, move_modify, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.repeat_ca = repeat_ca
        self.move_modify = move_modify

        self.suboperators = self.repeat_ca, self.move_modify

    def update(self, grid, action, per_env_context, shared_context, position, time):
        # Combine per-environment and shared context parameters

        grid, (next_per_env_context, time) = self.repeat_ca(
            grid, action, per_env_context, shared_context, time
        )

        grid, position = self.move_modify(grid, action, position)

        return grid, (next_per_env_context, position, time)
