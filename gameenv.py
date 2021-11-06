from globalconfig import gconfig
from gamedriver import SilentGameDriver2048
import numpy as np
from itertools import product



def encode_board(board, chanels=gconfig['CHANEL_ENCODED']):
    '''
    Transform board to state observation
    We will use one-hot encoded for each cell
    board: pure board representation
    chanels: length of one hot encoded vector
    return numpy array of shape 16x4x4
    '''
    state = np.zeros((chanels, 4, 4))
    for i, j in product(range(4), range(4)):
        cell = board[i][j]
        pos = int(np.log2(cell)) if cell > 0 else 0
        state[pos, i, j] = 1
    return state


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
        self.n_actions = 4

    def reset(self):
        '''
        Reset the game
        return the initial state of new game
        '''
        self.driver.restart() if self.driver.connected else self.driver.connect()
        self.score = 0
        return self.driver.get_board()


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
        r = (score - self.score - 4) * gconfig['REWARD_SCALE']        
        self.score = score
        return board, r, done


class EnvironementBatch:
    ''' 
    Batch of environment in A3C algorithm
    '''
    def __init__(self, n=gconfig['BATCH'], make_env=GameEnv, make_driver=SilentGameDriver2048):
        '''
        n: number of environments in batch
        make_env: function to make environment
        make_driver: function to make game driver
        '''
        self.envs = [make_env(make_driver()) for i in range(n)]

    def reset(self):
        '''
        Send reset signal to all environments in the batch
        return batch of first states after reset
        '''
        return np.array([env.reset() for env in self.envs])

    def step(self, actions):
        """
        Send actions to environments
        actions: list of actions corresponding to each environment
        return new_states, rewards, done
        """
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        new_obs, rewards, dones = map(np.array, zip(*results))
        # reset environments automatically
        for i in range(len(self.envs)):
            if dones[i]:
                new_obs[i] = self.envs[i].reset()
        return new_obs, rewards, dones