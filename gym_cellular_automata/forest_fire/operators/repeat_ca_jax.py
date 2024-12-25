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

    def update(self, grid, action, context):
        ca_params, accu_time = context

        time_action = self.t_acting(action)
        time_state = self.t_perception((grid, context))
        time_taken = time_action + time_state

        new_accu_time = accu_time + time_taken
        modf_accu_time, repeats = jnp.modf(new_accu_time)
        reshaped_repeats = repeats.reshape(()).astype(jnp.int32)

        def _ca_step(carry):
            grid, action, carry_ca_params = carry
            new_grid, new_ca_params = self.ca(grid, action, carry_ca_params)
            return (new_grid, action, new_ca_params)

        # ... in the main function ...
        grid, _, new_ca_params = lax.fori_loop(
            0,
            jnp.asarray(reshaped_repeats, dtype=jnp.int32),
            lambda i, carry: _ca_step(carry),
            (grid, action, ca_params),
        )

        return grid, (new_ca_params, modf_accu_time)
