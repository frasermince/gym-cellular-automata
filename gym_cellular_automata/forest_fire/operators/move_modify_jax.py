from typing import Dict, Set

import jax.numpy as jnp
from gymnasium import logger, spaces

from gym_cellular_automata.operator import Operator


class MoveJax(Operator):
    grid_dependant = (
        False  # If a constant size grid is used (that is usually the case).
    )
    action_dependant = True
    context_dependant = True

    deterministic = True

    def __init__(self, directions_sets: Dict[str, Set], *args, **kwargs):
        super().__init__(*args, **kwargs)

        # fmt: off
        self.up_set       = directions_sets["up"]
        self.down_set     = directions_sets["down"]
        self.left_set     = directions_sets["left"]
        self.right_set    = directions_sets["right"]
        self.not_move_set = directions_sets["not_move"]
        # fmt: on

        self.movement_set = (
            self.up_set
            | self.down_set
            | self.left_set
            | self.right_set
            | self.not_move_set
        )

    def update(self, grid, action, context):
        # A common input is a scalar of type ndarray
        # action = int(action)

        def get_new_position(position: tuple) -> jnp.array:
            row, col = position

            nrows, ncols = grid.shape

            # fmt: off
            valid_up    = row > 0
            valid_down  = row < (nrows - 1)
            valid_left  = col > 0
            valid_right = col < (ncols - 1)

            row = jnp.where(jnp.any(action == jnp.array(list(self.up_set))) & valid_up, row - 1, row)
            row = jnp.where(jnp.any(action == jnp.array(list(self.down_set))) & valid_down, row + 1, row)
            col = jnp.where(jnp.any(action == jnp.array(list(self.left_set))) & valid_left, col - 1, col)
            col = jnp.where(jnp.any(action == jnp.array(list(self.right_set))) & valid_right, col + 1, col)
            # fmt: on

            return jnp.array([row, col])

        return grid, get_new_position(context)


class ModifyJax(Operator):
    hit = False

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    deterministic = True

    def __init__(self, effects: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.effects = effects
        # Convert effects dict to arrays during initialization
        self.effect_keys = jnp.array(list(effects.keys()))
        self.effect_values = jnp.array(list(effects.values()))

    def update(self, grid, action, context):
        self.hit = False

        row, col = context

        cell_value = grid[row, col]

        matches = cell_value == self.effect_keys
        new_value = jnp.sum(matches * self.effect_values)

        # Only modify if action is True and we have an effect
        has_effect = jnp.any(matches)
        should_modify = action & has_effect

        grid = grid.at[row, col].set(
            jnp.where(should_modify, new_value, cell_value).reshape(())
        )

        return grid, context


class MoveModifyJax(Operator):
    grid_dependant = True
    action_dependant = True
    context_dependant = True

    deterministic = True

    def __init__(self, move, modify, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.suboperators = move, modify

        self.move = move
        self.modify = modify

        if self.action_space is None:
            if (
                self.move.action_space is not None
                and self.move.action_space is not None
            ):
                self.action_space = spaces.Tuple(
                    (self.move.action_space, self.move.action_space)
                )

        if self.context_space is None:
            if (
                self.move.context_space is not None
                and self.modify.context_space is not None
            ):
                assert self.move.context_space == self.modify.context_space
                self.context_space = self.move.context_space

    def update(self, grid, subactions, position):
        move_action = subactions[0]
        modify_action = subactions[1]

        grid, position = self.move(grid, move_action, position)
        grid, position = self.modify(grid, modify_action, position)

        return grid, position
