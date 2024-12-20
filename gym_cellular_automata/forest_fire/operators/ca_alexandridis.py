import numpy as np
from gymnasium import spaces

from gym_cellular_automata._config import TYPE_BOX
from gym_cellular_automata.forest_fire.utils.neighbors import neighborhood_at
from gym_cellular_automata.operator import Operator

import random
import math


def normalize_p(p):
    p = np.asarray(p).astype("float64")
    p = p / np.sum(p)
    return p


class PartiallyObservableForestFire(Operator):
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

    def update(self, grid, action, context):
        # A copy is needed for the sequential update of a CA
        wind_matrix = context["wind"]
        density_matrix = context["density"]
        vegetation_matrix = context["vegetation"]
        slope_matrix = context["slope"]

        new_grid = grid.copy()
        p_fire = context["p_fire"]
        p_tree = context["p_tree"]

        for row, cells in enumerate(grid):
            for col, cell in enumerate(cells):
                neighbors, neighbors_grid = neighborhood_at(
                    grid, (row, col), invariant=self.empty, return_grid=True
                )

                if cell == self.tree and self.fire in neighbors:
                    has_set_fire = False
                    for env_row in [0, 1, 2]:
                        if has_set_fire:
                            break
                        for env_col in [0, 1, 2]:
                            if has_set_fire:
                                break

                            if (
                                neighbors_grid[env_row][env_col] == self.fire
                            ):  # we only care there is a neighbour that is burning
                                p_veg = {1: -0.3, 2: 0, 3: 0.4}[
                                    vegetation_matrix[row][col]
                                ]
                                p_den = {1: -0.4, 2: 0, 3: 0.3}[
                                    density_matrix[row][col]
                                ]
                                p_h = 0.58
                                a = 0.078
                                slope = slope_matrix[row][col][env_row][env_col]
                                p_slope = np.exp(a * slope)
                                p_wind = wind_matrix[env_row][env_col]
                                p_burn = (
                                    p_h * (1 + p_veg) * (1 + p_den) * p_wind * p_slope
                                )
                                # print(
                                #     "p_veg p_den p_wind p_slope p_burn",
                                #     p_veg,
                                #     p_den,
                                #     p_wind,
                                #     p_slope,
                                #     p_burn,
                                # )
                                if p_burn > random.random():
                                    new_grid[row][col] = self.fire
                                    has_set_fire = True

                                    # Burn tree to the ground

                elif cell == self.tree:
                    # Sample for lightning strike

                    strike = self.np_random.choice(
                        [True, False], p=normalize_p([p_fire, 1 - p_fire])
                    )

                    new_grid[row][col] = self.fire if strike else cell

                elif cell == self.empty:
                    # Sample to grow a tree
                    growth = self.np_random.choice(
                        [True, False], p=normalize_p([p_tree, 1 - p_tree])
                    )

                    new_grid[row][col] = self.tree if growth else cell

                elif cell == self.fire:
                    # Consume fire
                    new_grid[row][col] = self.empty

        return new_grid, context
