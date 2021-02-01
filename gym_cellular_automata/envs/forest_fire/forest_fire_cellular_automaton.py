import numpy as np
from collections import namedtuple
from gym import spaces

from gym_cellular_automata import Grid

from gym_cellular_automata.operators import CellularAutomaton
from gym_cellular_automata.utils.neighbors import neighborhood_at

# ------------ Forest Fire Cellular Automaton

# Move to YAML
CELL_SYMBOLS = {
    'empty': 0,
    'tree': 1,
    'fire': 2
    }

UpdateSpaces = namedtuple('UpdateSpaces', ['grid_space', 'action_space', 'context_space'])

class ForestFireCellularAutomaton(CellularAutomaton):
    empty = CELL_SYMBOLS['empty']
    tree = CELL_SYMBOLS['tree']
    fire = CELL_SYMBOLS['fire']
    
    def __init__(self, grid_space=None, action_space=None, context_space=None):
        
        if context_space is None:
            context_space = spaces.Box(0.0, 1.0, shape=(2,))
        
        self.grid_space = grid_space
        self.action_space = action_space
        self.context_space = context_space

    def update(self, grid, action, context):
        # A copy is needed for the sequential update of a CA
        new_grid = Grid(grid.data.copy(), cell_states=3)
        p_fire, p_tree = context.data
        
        for row, cells in enumerate(grid.data):
            for col, cell in enumerate(cells):
                
                neighbors = neighborhood_at(grid, pos=(row, col), invariant=self.empty)
                
                if cell == self.tree and self.fire in neighbors:
                    # Burn tree to the ground
                    new_grid[row][col] = self.fire
                
                elif cell == self.tree:
                    # Sample for lightning strike
                    strike = np.random.choice([True, False], 1, p=[p_fire, 1-p_fire])[0]
                    new_grid[row][col] = self.fire if strike else cell
                
                elif cell == self.empty:
                    # Sample to grow a tree
                    growth = np.random.choice([True, False], 1, p=[p_tree, 1-p_tree])[0]
                    new_grid[row][col] = self.tree if growth else cell
                
                elif cell == self.fire:
                    # Consume fire
                    new_grid[row][col] = self.empty
                
                else:
                    continue
                   
        return new_grid, context
