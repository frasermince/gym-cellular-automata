from typing import Optional, Tuple

import jax.numpy as jnp
import jax
import numpy as np
from jax import jit
from gymnasium import spaces

from gym_cellular_automata._config import TYPE_BOX, TYPE_INT
from gym_cellular_automata.ca_env import CAEnv
from gym_cellular_automata.forest_fire.operators import (
    ModifyJax,
    MoveJax,
    MoveModifyJax,
    RepeatCAJax,
    PartiallyObservableForestFireJax,
)
from gym_cellular_automata.grid_space import GridSpace
from gym_cellular_automata.operator import Operator

from .utils.advanced_bulldozer_render import render, plot_grid_attribute
from .utils.init_utils import (
    init_vegetation,
    init_density,
    init_altitude,
    init_density_same,
    init_vegetation_same,
    init_altitude_same,
    create_up_to_k_mappings,
    get_winds,
    get_slope,
)
from .utils.extension_utils import (
    transform_grid,
    apply_extensions,
    EXTENSION_REGISTRY,
)

import math
from functools import partial


class AdvancedForestFireBulldozerEnv(CAEnv):
    metadata = {"render_modes": ["human"], "render_mode": "rgb_array"}

    @property
    def MDP(self):
        return self._MDP

    @property
    def initial_state(self):
        self.grid, fire_age = self._initial_grid_distribution()
        self.context = self._initial_context_distribution(fire_age, self.grid)

        initial_state = self.grid, self.context

        self._resample_initial = False

        return initial_state

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
        enable_extensions: bool = False,
        **kwargs,
    ):
        super().__init__(nrows, ncols, **kwargs)
        self.middle_fire = middle_fire
        self.use_hidden = use_hidden
        self.starting_key = key
        self.transform_grid = len(EXTENSION_REGISTRY) > 0

        self.shared_context_keys = {
            "winds",  # Shared wind patterns
            "p_fire",  # Global probability parameters
            "p_tree",
            "p_wind_change",
            "day_length",
        }

        self.per_env_context_keys = {
            "wind_index",
            "density",
            "vegetation",
            "altitude",
            "slope",
            "fire_age",
            "key",
            "is_night",
            "current_day_length",
            "true_grid",
            "time_step",
        }

        self.extension_map = [
            "unblur",
            "wind",
            "density",
            "vegetation",
            "altitude",
            "zoom",
        ]
        self._reward_per_empty = 0.0
        self._reward_per_tree = 1.0
        self._reward_per_fire = -1.0
        self._reward_per_bulldozed = 0.0

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
        self._bulldozed = 10  # Bulldozed cell

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
        self._p_tree = 0.0005
        self._p_wind_change = 0.06

        self._effects = {
            # self._fire: self._bulldozed,
            self._tree: self._bulldozed,
        }  # Substitution Effect

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
            self._empty, self._tree, self._fire, self._bulldozed, **self.ca_space
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
        self._MDP = MDP(
            self.repeater,
            self.move_modify,
            self.transform_grid and enable_extensions,
            enable_extensions,
            **self.MDP_space,
        )

    # def extension_lookups(self):
    #     self._extension_lookups

    def _create_full_actions(self, action):
        """Create full actions by combining base actions with binary extension actions"""
        non_comb_actions = self.action_space.shape[-1]
        binary_actions = []

        # Convert extension choices to binary actions
        for i, extension_lookup in enumerate(self._extension_lookups):
            extension_choice = action[:, non_comb_actions + i]
            binary_action = jnp.take(extension_lookup, extension_choice, axis=0)
            binary_actions.append(binary_action)

        # Concatenate binary actions if they exist
        if binary_actions:
            binary_actions = jnp.concatenate(binary_actions, axis=-1)

        # Get base actions
        full_actions = action[:, :non_comb_actions]

        # Combine with binary actions if they exist
        if len(binary_actions) > 0:
            full_actions = jnp.concatenate((full_actions, binary_actions), axis=-1)

        return full_actions

    @partial(jax.jit, static_argnums=0)
    def stateless_step(self, action, obs, info):
        # MDP Transition
        grid, context = obs

        # Use true grid from context if available, otherwise use grid
        true_grid = context["per_env_context"]["true_grid"]

        shared_context = context["shared_context"]
        per_env_context = context["per_env_context"]
        per_env_in_axes = {k: 0 for k in self.per_env_context_keys}

        # Create full actions including extensions
        full_actions = self._create_full_actions(action)

        (next_extended_grid, next_true_grid), updated_context = jax.vmap(
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
            true_grid,
            full_actions,
            per_env_context,
            shared_context,
            context["position"],
            context["time"],
        )
        per_env_portion, next_position, next_time = updated_context
        context["per_env_context"] = per_env_portion
        context["position"] = next_position
        context["time"] = next_time

        next_state = (next_extended_grid, context)

        # Check for termination
        next_done = jax.vmap(self._is_done, in_axes=0)(next_true_grid)

        # Gym API Formatting
        obs = next_state
        reward = jax.vmap(self._award, in_axes=(0, 0, 0))(
            true_grid, next_true_grid, per_env_context
        )
        terminated = next_done
        truncated = jnp.full(next_true_grid.shape[0], False)
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
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._resample_initial = True
        initial_grid, initial_context = self.initial_state

        obs = (initial_grid, initial_context)
        info = {
            "TimeLimit.truncated": jnp.full(initial_grid.shape[0], False),
            "terminated": jnp.full(initial_grid.shape[0], False),
            "steps_elapsed": jnp.zeros(initial_grid.shape[0]),
            "reward_accumulated": jnp.zeros(initial_grid.shape[0]),
            "reward": jnp.zeros(initial_grid.shape[0]),
        }

        return obs, info

    # @partial(jax.jit, static_argnums=(0,), static_argnames=("seed", "options"))
    def conditional_reset(
        self,
        step_tuple,
        action,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        def reset_fn(args):
            step_tuple, (initial_grid, initial_context) = args
            obs, reward, terminated, truncated, info = step_tuple
            grid, context = obs
            grid_obs_only = jnp.where(
                terminated[:, None, None], initial_grid[:, :, :, 0], grid[:, :, :, 0]
            )

            shared_context = context["shared_context"]
            per_env_context = context["per_env_context"]
            per_env_in_axes = {k: 0 for k in self.per_env_context_keys}

            # Create full actions including extensions
            full_actions = self._create_full_actions(action)

            next_grid = jax.vmap(
                lambda terminated, grid_obs_only, original_grid, position, action, per_env_context, shared_context: jnp.where(
                    terminated,
                    self.MDP.build_observation_on_extensions(
                        grid_obs_only, position, action, per_env_context, shared_context
                    ),
                    original_grid,
                ),
                in_axes=(
                    0,
                    0,
                    0,
                    0,
                    0,
                    per_env_in_axes,
                    None,
                ),  # grid, action, per_env_context, shared_context
            )(
                terminated,
                grid_obs_only,
                grid,
                context["position"],
                full_actions,
                per_env_context,
                shared_context,
            )

            for key in self.per_env_context_keys:
                key_shape = context["per_env_context"][key].shape
                expanded_terminated = terminated[
                    (...,) + (None,) * (len(key_shape) - 1)
                ]
                context["per_env_context"][key] = jnp.where(
                    expanded_terminated,
                    initial_context["per_env_context"][key],
                    context["per_env_context"][key],
                )
            obs = (next_grid, context)
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

    def _award(self, prev_grid, grid, per_env_context):
        #     """Reward Function

        #     Combines negative ratio of burning area with time pressure:
        #     raw_reward = -(f / (t + f)) - (time_step / max_steps)
        #     final_reward = tanh(raw_reward)

        #     Where:
        #         t: tree cell counts
        #         f: fire cell counts
        #         time_step: current timestep
        #         max_steps: normalization factor (e.g. 200)

        #     Both components naturally fall in [-1,0], so their sum is in [-2,0].
        #     Tanh maps this smoothly to [-1,1] with good gradients in the typical range.
        #     """
        counts = self.count_cells(grid)

        # Basic fire/tree ratio component
        t = counts[self._tree]
        f = counts[self._fire]
        base_reward = jnp.where(t + f > 0, -(f / (t + f)), 0.0)

        # Time pressure component (normalized to [0,1])
        time_step = per_env_context["time_step"]
        time_penalty = jnp.minimum(
            time_step / 100.0, 4.0
        )  # Caps at -1.0 after 200 steps

        # Combine and map to [-1,1]
        raw_reward = 1 + base_reward - time_penalty
        return jnp.tanh(raw_reward)

    # def _award(self, prev_grid, grid, per_env_context):
    #     ncells = grid.shape[0] * grid.shape[1]

    #     dict_counts = self.count_cells(grid)

    #     cell_counts = jnp.array(
    #         [dict_counts[self._empty], dict_counts[self._tree], dict_counts[self._fire]]
    #     )

    #     cell_counts_relative = cell_counts / ncells

    #     reward_weights = jnp.array(
    #         [self._reward_per_empty, self._reward_per_tree, self._reward_per_fire]
    #     )

    #     return jnp.dot(reward_weights, cell_counts_relative)

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

        total_extensions = sum(n for n, _ in self.extension_choices)
        grid_space = GridSpace(
            values = [  self._empty,   self._tree,   self._fire],
            probs  = [self._p_empty_init, self._p_tree_init,          0.0],
            shape=(self.num_envs, self.nrows, self.ncols, total_extensions + 3), # +3 for grid, position, and day night
        )
        # fmt: on

        grid = jnp.array(grid_space.sample(), dtype=jnp.float32)

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

        fire_age = self._fire_age
        for env in range(self.num_envs):
            r, c = self._pos_fire[env]
            grid = grid.at[env, r, c].set(self._fire)
            fire_age = fire_age.at[env, r, c].set(10)

        return jnp.array(grid), fire_age

    def _initial_context_distribution(self, fire_age, grid):
        init_time = jnp.zeros(self.num_envs)

        if self._pos_bull is None:
            # Bulldozer Position for each env
            # Fixed position 15% from upper right corner
            self._pos_bull = []
            for _ in range(self.num_envs):
                r = int(self.nrows * 0.15)  # 15% down from top
                c = int(self.ncols * 0.85)  # 15% in from right
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
            "fire_age": fire_age,
            "key": new_keys,
            "is_night": jnp.zeros(self.num_envs, dtype=jnp.int32),
            "current_day_length": jnp.zeros(self.num_envs, dtype=jnp.int32),
            "true_grid": grid[..., 0],
            "time_step": jnp.zeros(self.num_envs, dtype=jnp.int32),
        }

        # Shared context parameters
        shared_context = {
            "winds": self._winds,
            "p_fire": jnp.array(self._p_fire, dtype=jnp.float32),
            "p_tree": jnp.array(self._p_tree, dtype=jnp.float32),
            "p_wind_change": jnp.array(self._p_wind_change, dtype=jnp.float32),
            "day_length": 100,
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

        self._movement_timings[self._moves["not_move"]] = self._t_act_move
        self._shooting_timings[self._shoots["none"]] = self._t_act_shoot
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

    def _set_spaces(self):
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
            "current_day_length": spaces.Box(
                0,
                float("inf"),
                shape=(self.num_envs,),
                dtype=TYPE_BOX,
            ),
            "is_night": spaces.Box(
                0,
                1,
                shape=(self.num_envs,),
                dtype=TYPE_BOX,
            ),
            "time_step": spaces.Box(
                0, float("inf"), shape=(self.num_envs,), dtype=TYPE_BOX
            ),
            "true_grid": spaces.Box(
                0, 2, shape=(self.num_envs, self.nrows, self.ncols), dtype=TYPE_BOX
            ),
            # "key": spaces.Box(0, float("inf"), shape=(self.num_envs,), dtype=TYPE_BOX),
        }

        # Create spaces for shared context parameters
        self.shared_context_space = {
            "winds": spaces.Box(0.0, 1.0, shape=(8, 3, 3), dtype=TYPE_BOX),
            "p_fire": spaces.Box(0.0, 1.0, shape=(), dtype=TYPE_BOX),
            "p_tree": spaces.Box(0.0, 1.0, shape=(), dtype=TYPE_BOX),
            "p_wind_change": spaces.Box(0.0, 1.0, shape=(), dtype=TYPE_BOX),
            "day_length": spaces.Box(0.0, float("inf"), shape=(), dtype=TYPE_BOX),
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
        m, n = len(self._moves), len(self._shoots)
        self.action_space = spaces.MultiDiscrete(
            nvec=np.array([[m, n]] * self.num_envs),  # Shape becomes (num_envs, 2)
            dtype=TYPE_INT,
        )

        # Create extension choices based on registry configuration
        self.extension_choices = []
        for reg in EXTENSION_REGISTRY:
            n = len(reg.extensions)  # number of extensions
            k = reg.choose  # how many can be chosen
            self.extension_choices.append((n, k))

        # Combined action space with extensions
        extension_nvec = np.array(
            [
                sum(math.comb(n, i) for i in range(k + 1))
                for n, k in self.extension_choices
            ]
        )
        action_nvec = np.array([m, n])

        self.extension_space = spaces.MultiDiscrete(
            nvec=np.array([math.comb(n, k) for n, k in self.extension_choices]),
            dtype=TYPE_INT,
        )

        self.total_action_space = spaces.MultiDiscrete(
            nvec=[np.concatenate([action_nvec, extension_nvec])] * self.num_envs,
            dtype=TYPE_INT,
        )
        total_extensions = sum(n for n, _ in self.extension_choices)
        self.grid_space = GridSpace(
            values=[self._empty, self._tree, self._fire],
            shape=(
                self.num_envs,
                self.nrows,
                self.ncols,
                total_extensions + 3,
            ),  # +3 for grid, position, and day night
        )

        # Create mappings between combinatorial IDs and individual indices
        self._extension_lookups = []
        for n, k in self.extension_choices:
            # Generate mappings for combinations up to size k
            id_to_binary, _ = create_up_to_k_mappings(n, k)
            self._extension_lookups.append(id_to_binary)

        # Combined action space with extensions

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
            self._bulldozed: jnp.sum(grid == self._bulldozed),
        }

        return counts


class MDP(Operator):
    grid_dependant = True
    action_dependant = True
    context_dependant = True

    deterministic = False

    def __init__(
        self,
        repeat_ca,
        move_modify,
        should_transform_grid,
        enable_extensions,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.repeat_ca = repeat_ca
        self.move_modify = move_modify
        self.should_transform_grid = should_transform_grid
        self.enable_extensions = enable_extensions
        self.suboperators = self.repeat_ca, self.move_modify

    # Gym API
    # step, reset & seed methods inherited from parent class
    @partial(jax.jit, static_argnums=(0))
    def build_observation_on_extensions(
        self,
        grid,
        position,
        actions,
        per_env_context,
        shared_context,
    ):
        # Base observation with all transformations
        if self.should_transform_grid:
            transformed_grid = transform_grid(grid, per_env_context, 0, 0)
        else:
            transformed_grid = grid

        # Base channels
        channels = [transformed_grid]

        # Position channel
        x_pos, y_pos = position[..., 0], position[..., 1]
        pos_channel = jnp.zeros_like(grid)
        pos_channel = pos_channel.at[x_pos, y_pos].set(1)
        day_night_channel = jnp.where(
            per_env_context["is_night"], jnp.ones_like(grid), jnp.zeros_like(grid)
        )
        channels.append(pos_channel)
        channels.append(day_night_channel)

        # Extension channels
        extension_channels = apply_extensions(
            grid, actions, per_env_context, shared_context, self.enable_extensions
        )
        channels = jnp.stack([*channels, *extension_channels], axis=-1)

        return channels

    def update(self, grid, action, per_env_context, shared_context, position, time):
        # Combine per-environment and shared context parameters
        basic_action = (action[0], action[1])

        grid, (next_per_env_context, next_time) = self.repeat_ca(
            grid, basic_action, per_env_context, shared_context, time
        )

        grid, position = self.move_modify(grid, basic_action, position)

        # Store true grid state
        next_per_env_context["true_grid"] = grid
        next_per_env_context["time_step"] = next_per_env_context["time_step"] + 1
        extended_grid = self.build_observation_on_extensions(
            grid, position, action, per_env_context, shared_context
        )

        return (extended_grid, grid), (next_per_env_context, position, next_time)
