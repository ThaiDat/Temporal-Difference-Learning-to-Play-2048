from globalconfig import gconfig


class GameEnv:
    '''
    Interface bridge the agent and the game
    '''

    def __init__(self, driver):
        self.driver = driver


    def reset(self):
        '''
        Reset the game
        return the initial state of new game
        '''
        self.driver.restart() if self.driver.connected else self.driver.connect()
        return self.driver.get_board()


    def step(self, action):
        '''
        React to action from the agent
        we design reward as follow
        return (observation after doing action, reward from action, is terminal state)
        '''
        pass