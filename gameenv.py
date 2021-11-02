from globalconfig import gconfig


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
        We design reward as score changed get from the real game
        action: action the agent sent to the environment
        return (observation after doing action, reward from action, is terminal state)
        '''
        self.driver.make_move(action)
        s = self.driver.get_board()
        score = self.driver.get_score()
        done = self.driver.is_end()
        r = (score - self.score) * gconfig['REWARD_SCALE']
        self.score = score
        return s, r, done
