import math
from typing import Callable

import numpy as np

from gym_cellular_automata._config import TYPE_BOX
from gym_cellular_automata.operator import Operator
import jax.numpy as jnp
import jax.lax as lax


class RepeatCAJax(Operator):
    grid_dependant = True
    action_dependant = True
    context_dependant = True

    def __init__(
        self,
        cellular_automaton,
        t_acting: Callable,
        t_perception: Callable,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.t_acting = t_acting
        self.t_perception = t_perception

        self.ca = cellular_automaton
        self.suboperators = (self.ca,)
        self.deterministic = self.ca.deterministic

    def update(self, grid, action, per_env_context, shared_context, accu_time):
        time_action = self.t_acting(action)
        time_state = self.t_perception((grid, per_env_context, shared_context))
        time_taken = time_action + time_state

        new_accu_time = accu_time + time_taken
        modf_accu_time, repeats = jnp.modf(new_accu_time)
        reshaped_repeats = repeats.reshape(()).astype(jnp.int32)

        def _ca_step(carry):
            grid, action, carry_per_env, carry_shared = carry
            new_grid, new_per_env, new_shared = self.ca(
                grid, action, carry_per_env, carry_shared
            )
            return (new_grid, action, new_per_env, new_shared)

        # ... in the main function ...
        # jax.debug.callback(
        #     repeats_printer,
        #     (
        #         jnp.array(
        #             [
        #                 reshaped_repeats,
        #             ]
        #         )
        #     ),
        # )
        grid, _, new_per_env, _ = _ca_step(
            (grid, action, per_env_context, shared_context)
        )
        # grid, _, new_per_env, _ = lax.fori_loop(
        #     0,
        #     jnp.asarray(reshaped_repeats, dtype=jnp.int32),
        #     lambda i, carry: _ca_step(carry),
        #     (grid, action, per_env_context, shared_context),
        # )

        return grid, (new_per_env, modf_accu_time)
