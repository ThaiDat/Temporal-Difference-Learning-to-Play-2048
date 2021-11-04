from globalconfig import gconfig
import numpy as np
from itertools import product


class GameEnv:
    '''
    Interface bridge the agent and the game
    '''

    def __init__(self, driver):
        '''
        driver: game driver to attach to the environment. The environment will use this driver to interact with the game
        '''
        self.driver = driver
        self.score = 0
        self.state_space = (gconfig['CHANEL_ENCODED'], 4, 4)
        self.n_actions = 4


    def reset(self):
        '''
        Reset the game
        return the initial state of new game
        '''
        self.driver.restart() if self.driver.connected else self.driver.connect()
        self.score = 0
        return self.__process_board(self.driver.get_board())


    def step(self, action):
        '''
        React to action from the agent
        We design reward as score changed get from the real game. we also purnish invalid moves (self loop)
        action: action the agent sent to the environment
        return (observation after doing action, reward from action, is terminal state)
        '''
        # Make action and retrieve information
        self.driver.make_move(action)
        board = self.driver.get_board()
        score = self.driver.get_score()
        done = self.driver.is_end()
        # Process information
        s = self.__process_board(board)
        r = (score - self.score - 4) * gconfig['REWARD_SCALE']        
        self.score = score
        return s, r, done

    def __process_board(self, board):
        '''
        Transform board to state observation
        We will use one-hot encoded for each cell
        return numpy array of shape 16x4x4
        '''
        state = np.zeros(self.state_space)
        for i, j in product(range(4), range(4)):
            cell = board[i][j]
            pos = pos = int(np.log2(cell)) if cell > 0 else 0
            state[pos, i, j] = 1
        return state
