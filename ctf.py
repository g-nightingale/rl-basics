import numpy as np

class SimpleCTF:
    """
    A very simple capture the flag environment.
    """
    def __init__(self):
        self.grid_size = 4
        self.flag_pos = (0, 3)
        self.home_pos = (3, 0)
        self.block_tiles_pos = [(2, 2), (2, 3)]
        self.player_tile = 1
        self.block_tile = 2
        self.flag_tile = 3
        self.step_reward = -1
        self.terminal_reward = 10
        self.reset()

    def reset(self):
        start_x = np.random.randint(0, self.grid_size)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        for pos in self.block_tiles_pos:
            self.grid[pos] = self.block_tile
        self.grid[self.flag_pos] = self.flag_tile
        
        self.grid[3, start_x] = self.player_tile
        self.state = (3, start_x, 0)
        self.terminal = False

    def move(self, action):
        """
        Move the agent
            Actions:
                0: Up
                1: Down
                2: Left
                3: Right
        """
        reward = self.step_reward

        y = self.state[0]
        x = self.state[1]
        flag_held = self.state[2]

        action_0_pos = (max(y - 1, 0), x)
        action_1_pos = (min(y + 1, self.grid_size - 1), x)
        action_2_pos = (y, max(x - 1, 0))
        action_3_pos = (y, min(x + 1, self.grid_size - 1))

        curr_pos = (y, x)
        new_pos = (y, x)
        if action == 0 and action_0_pos not in self.block_tiles_pos:
            new_pos = action_0_pos
        if action == 1 and action_1_pos not in self.block_tiles_pos:
            new_pos = action_1_pos
        if action == 2 and action_2_pos not in self.block_tiles_pos:
            new_pos = action_2_pos
        if action == 3 and action_3_pos not in self.block_tiles_pos:
            new_pos = action_3_pos

        if new_pos == self.flag_pos and flag_held == 0:
            flag_held = 1
            self.grid[self.flag_pos] = 0

        if new_pos == self.home_pos and flag_held == 1:
            reward += self.terminal_reward
            self.terminal = True

        self.grid[curr_pos] = 0
        self.grid[new_pos] = self.player_tile

        self.state = (new_pos[0], new_pos[1], flag_held)

        return reward

    def step(self, action):
        """
        Step the environment.
        """
        reward = self.move(action)
        return self.state, reward, self.terminal
    
    def render(self):
        print(self.state, '\n', self.grid, '\n')