from globalconfig import gconfig
from abc import abstractmethod
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains

class GameDriver2048:
    '''
    Base class for game driver 2048
    This class can be derived into several versions: interactive web game, local game with graphic,
        local game with console, local game without console and more, …
    '''
    @abstractmethod
    def connect(self):
        '''Initialize and connect to real game'''
        pass

    @abstractmethod
    def get_score(self):
        '''
        Get the current score of the game
        '''
        pass

    @abstractmethod
    def get_board(self):
        '''Get current board state'''
        pass

    @abstractmethod
    def make_move(self, move):
        '''Send move signal to game'''
        pass

    @abstractmethod
    def restart(self):
        '''Restart the game'''
        pass

    @abstractmethod
    def is_end(self):
        '''Check gameover'''
        pass


class WebGameDriver2048(GameDriver2048):
    '''
    Interface for playing 2048 on the web
    '''

    def __init__(self, gameurl=gconfig['GAME_URL'], browser=gconfig['BROWSER'], driver_file=gconfig['BROWSER_DRIVER_FILE']):
        self.connected = False
        self.url = gameurl
        self.driver = getattr(webdriver, browser)(executable_path=driver_file)
        self.driver.set_window_position(*gconfig['BROWSER_POSITION'])
        self.driver.set_window_size(*gconfig['BROWSER_SIZE'])

    def connect(self):
        '''
        Connect with the game on browser. 
        '''
        self.driver.get(self.url)
        self.actor = ActionChains(self.driver)
        self.__get_elements()
        self.connected = True

    @abstractmethod
    def get_score(self):
        '''
        Get the current score of the game
        '''
        # score_nums = [current score, bonus score]
        score_nums = self.__extract_positive_number(self.score.text)
        return 0 if len(score_nums) == 0 else score_nums[0]

    def get_board(self):
        '''
        Get the current board state
        return the 
        '''
        # class attribute contains infos we need: tile value, tile column, tile row
        # There still some merged/eliminated tiles, but it should be overrided when assigned to board
        tiles = self.tile_container.find_elements(By.CLASS_NAME, 'tile')
        tiles = [t.get_attribute('class') for t in tiles]
        tiles = map(self.__extract_positive_number, tiles)
        board = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
        for value, col, row in tiles:
            # row and col on the web is from 1 
            board[row-1][col-1] = value
        return board

    def make_move(self, move, sleep=gconfig['ACTION_SLEEP']):
        '''
        Send move signal to web browser
        move: 0=left, 1=up, 2=right, 3=down
        '''
        move = gconfig['ACTION_MAP'][move] if isinstance(move, int) else str(move).upper()
        self.actor.send_keys(getattr(Keys, move)).perform()
        time.sleep(sleep)

    def restart(self):
        '''Restart the game'''
        self.restart_button.click()
        self.__get_elements()

    def is_end(self):
        '''
        Check gameover
        return True if gameover
        '''
        return self.game_message.get_attribute('class').endswith('game-over')
    
    def __extract_positive_number(self, s):
        '''
        Helper function that extract all positive number from string (exclude negative sign and 0)
        s: input string
        n: maximum numbers to extract. This function will return after discover first n numbers
        return list of numbers (len <= n)
        '''
        nums = list(); num = 0
        for c in s:
            if c.isdigit():
                num = num * 10 + int(c)
            elif num > 0:
                nums.append(num)
                num = 0
        # Last append
        if num > 0:
            nums.append(num)
        return nums

    def __get_elements(self):
        '''
        Get the necessary elements. Used when a game start/restart
        '''
        container = self.driver.find_element(By.CLASS_NAME, 'container')
        self.tile_container = container.find_element(By.CLASS_NAME, 'tile-container')
        self.game_message = container.find_element(By.CLASS_NAME, 'game-message')
        self.restart_button = container.find_element(By.CLASS_NAME, 'restart-button')
        self.score = container.find_element(By.CLASS_NAME, 'score-container')
        