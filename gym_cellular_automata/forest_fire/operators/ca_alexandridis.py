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

    def sample_pinecone_parameters(self, directions=8):
        # Sample number of pinecones for each cell from Poisson distribution
        N_p = self.np_random.poisson()

        # Sample directions for each pinecone
        total_pinecones = np.sum(N_p)

        # Generate random integers 0-7 representing directions
        # Direction indices in 3x3 grid:
        # [0,1,2]
        # [3,4,5] where 4 is center
        # [6,7,8]
        directions = self.np_random.integers(0, 8, size=total_pinecones)

        # 3x3 lookup grid for wind directions, excluding center (1,1)
        lookup_grid = [
            (0, 0),
            (0, 1),
            (0, 2),  # Top row
            (1, 0),
            (1, 2),  # Middle row (skip center)
            (2, 0),
            (2, 1),
            (2, 2),  # Bottom row
        ]
        grid_indices = np.array([lookup_grid[d] for d in directions])

        # Convert to (dx, dy) grid movements using lookup tables
        dx_lookup = [1, 1, 0, -1, -1, -1, 0, 1]  # E, NE, N, NW, W, SW, S, SE
        dy_lookup = [0, 1, 1, 1, 0, -1, -1, -1]  # E, NE, N, NW, W, SW, S, SE

        dx = np.array([dx_lookup[d] for d in directions])
        dy = np.array([dy_lookup[d] for d in directions])

        return N_p, (dx, dy, grid_indices)

    def _set_fire(
        self,
        neighbors_grid,
        row,
        col,
        new_grid,
        wind_matrix,
        density_matrix,
        vegetation_matrix,
        slope_matrix,
        altitude_matrix,
        fire_age,
    ):
        # Get altitude values in 3x3 neighborhood centered on current cell
        # row_start = max(0, row - 1)
        # row_end = min(altitude_matrix.shape[0], row + 2)
        # col_start = max(0, col - 1)
        # col_end = min(altitude_matrix.shape[1], col + 2)

        # altitude_neighborhood = altitude_matrix[row_start:row_end, col_start:col_end]
        fire_grid = neighbors_grid == self.fire
        p_veg = {1: -0.3, 2: 0.0, 3: 0.3, 4: 0.6, 5: 1.0}[vegetation_matrix[row][col]]
        p_den = {1: -0.4, 2: 0, 3: 0.3, 4: 0.6, 5: 1.0}[density_matrix[row][col]]
        p_h = 0.58
        a = 0.078

        slope = slope_matrix[row][col]
        p_slope = np.exp(a * slope)
        p_burn = p_h * (1 + p_veg) * (1 + p_den) * wind_matrix * p_slope
        print("p_burn min max", np.min(p_burn), np.max(p_burn))
        # import pdb

        # pdb.set_trace()
        burn_probability = p_burn > self.np_random.uniform(0, 1, p_burn.shape)
        # import pdb

        # pdb.set_trace()

        if np.any(fire_grid & burn_probability):
            new_grid[row][col] = self.fire
            fire_age[row][col] = self.np_random.integers(4, 11)

    def _set_fire_pinecone(
        self,
        row,
        col,
        new_grid,
        density_matrix,
        vegetation_matrix,
        fire_age,
    ):
        p_veg = {1: 0.0, 2: 0.8, 3: 1.6, 4: 2.0, 5: 2.5}[vegetation_matrix[row][col]]
        p_den = {1: 0.0, 2: 0.6, 3: 1.2, 4: 1.5, 5: 2.0}[density_matrix[row][col]]
        p_h = 0.58

        p_burn = p_h * (1 + p_veg) * (1 + p_den)
        burn_probability = p_burn > self.np_random.uniform(0, 1)

        if np.any(burn_probability):
            new_grid[row][col] = self.fire
            fire_age[row][col] = self.np_random.integers(4, 11)
            return True
        return False

    def update(self, grid, action, context):
        # A copy is needed for the sequential update of a CA
        wind_matrix, ft = context["winds"][context["wind_index"]]
        density_matrix = context["density"]
        vegetation_matrix = context["vegetation"]
        slope_matrix = context["slope"]
        altitude_matrix = context["altitude"]

        new_grid = grid.copy()
        p_tree = context["p_tree"]
        p_wind_change = context["p_wind_change"]
        fire_age = context["fire_age"]
        skipped_indices = []

        for row, cells in enumerate(grid):
            for col, cell in enumerate(cells):
                if (row, col) in skipped_indices:
                    continue
                neighbors, neighbors_grid = neighborhood_at(
                    grid, (row, col), invariant=self.empty, return_grid=True
                )

                if cell == self.tree and self.fire in neighbors:
                    self._set_fire(
                        neighbors_grid,
                        row,
                        col,
                        new_grid,
                        wind_matrix,
                        density_matrix,
                        vegetation_matrix,
                        slope_matrix,
                        altitude_matrix,
                        fire_age,
                    )

                elif cell == self.empty:
                    # Sample to grow a tree
                    growth = self.np_random.choice(
                        [True, False], p=normalize_p([p_tree, 1 - p_tree])
                    )

                    new_grid[row][col] = self.tree if growth else cell

                elif cell == self.fire:
                    # Consume fire
                    fire_age[row][col] -= 1
                    if fire_age[row][col] == 0:
                        new_grid[row][col] = self.empty
                    number_pinecones, (dx, dy, grid_indices) = (
                        self.sample_pinecone_parameters()
                    )
                    if number_pinecones == 0:
                        continue
                    pinecone_thrust = 3 * np.random.standard_normal(number_pinecones)
                    pinecone_thrust = pinecone_thrust * ft[tuple(zip(*grid_indices))]
                    for i in range(number_pinecones):
                        new_row = round(row + dx[i] * pinecone_thrust[i])
                        new_col = round(col + dy[i] * pinecone_thrust[i])
                        if (
                            new_row >= 0
                            and new_row < len(grid)
                            and new_col >= 0
                            and new_col < len(grid[0])
                            and (new_row, new_col) != (row, col)
                        ):
                            did_burn = self._set_fire_pinecone(
                                new_row,
                                new_col,
                                new_grid,
                                density_matrix,
                                vegetation_matrix,
                                fire_age,
                            )
                            if did_burn:
                                skipped_indices.append((new_row, new_col))

        wind_change = self.np_random.choice(
            [True, False], p=normalize_p([p_wind_change, 1 - p_wind_change])
        )
        if wind_change:
            new_wind = self.np_random.integers(1, 8)  # Sample between 1 and 7
            context["wind_index"] = (context["wind_index"] + new_wind) % len(
                context["winds"]
            )
            print(f"Wind change: {new_wind} steps")
        return new_grid, context
