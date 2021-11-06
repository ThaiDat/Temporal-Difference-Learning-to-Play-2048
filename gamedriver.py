from globalconfig import gconfig
from abc import abstractmethod
import time
from random import choice, random
from itertools import product
# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.action_chains import ActionChains


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
        '''
        gameurl: web url to play game.
        browser: Name of the browser used to play the game
        driver_file: path to the driver to control the browser
        If the developer do not change, they will use the value in global config by default
        '''
        super(WebGameDriver2048, self).__init__()
        self.connected = False
        self.url = gameurl
        self.driver = getattr(webdriver, browser)(executable_path=driver_file)

    def connect(self):
        '''
        Connect with the game on browser. 
        '''
        self.driver.get(self.url)
        self.actor = ActionChains(self.driver)
        self.__get_elements()
        self.connected = True

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
        return the board 4x4
        '''
        # class attribute contains infos we need: tile value, tile column, tile row
        # There still some merged/eliminated tiles, but it should be overrided when assigned to board
        tiles = self.tile_container.find_elements(By.CLASS_NAME, 'tile')
        tiles = [t.get_attribute('class') for t in tiles]
        tiles = map(self.__extract_positive_number, tiles)
        board = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        for value, col, row in tiles:
            # row and col on the web is from 1
            board[row-1][col-1] = value
        return board

    def make_move(self, move, sleep=gconfig['ACTION_SLEEP']):
        '''
        Send move signal to web browser
        move: 0=left, 1=up, 2=right, 3=down
        '''
        move = gconfig['ACTION_MAP'][move] if isinstance(
            move, int) else str(move).upper()
        self.actor.send_keys(getattr(Keys, move)).perform()
        time.sleep(sleep)

    def restart(self):
        '''Restart the game'''
        self.restart_button.click()
        time.sleep(gconfig['ACTION_SLEEP'])
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
        nums = list()
        num = 0
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
        self.tile_container = container.find_element(
            By.CLASS_NAME, 'tile-container')
        self.game_message = container.find_element(
            By.CLASS_NAME, 'game-message')
        self.restart_button = container.find_element(
            By.CLASS_NAME, 'restart-button')
        self.score = container.find_element(By.CLASS_NAME, 'score-container')


class SilentGameDriver2048:
    '''
    Interface for playing 2048 silently
    '''

    def __init__(self):
        super(SilentGameDriver2048, self).__init__()
        self.connected = False

    def connect(self):
        '''Initialize and connect to real game'''
        self.restart()
        self.connected = True

    def get_score(self):
        '''
        Get the current score of the game
        '''
        return self.score

    def get_board(self):
        '''
        Get the current board state
        return the board 4x4
        '''
        # return the copy of the board
        return [row[:] for row in self.board]

    def make_move(self, move):
        '''Send move signal to game'''
        # direct
        iterator = None; jterator = None
        di = 0; dj = 0
        if move == 0 or move == gconfig['ACTION_MAP'][0]:  # left
            iterator = range(0, 4)
            jterator = range(0, 4)
            di = 0
            dj = -1
        elif move == 1 or move == gconfig['ACTION_MAP'][1]:  # up
            iterator = range(0, 4)
            jterator = range(0, 4)
            di = -1
            dj = 0
        elif move == 2 or move == gconfig['ACTION_MAP'][2]:  # right
            iterator = range(0, 4)
            jterator = range(3, -1, -1)
            di = 0
            dj = 1
        elif move == 3 or move == gconfig['ACTION_MAP'][3]:  # down
            iterator = range(3, -1, -1)
            jterator = range(0, 4)
            di = 1
            dj = 0
        # move
        modified = False
        left_walls = [-1] * 4
        top_walls = [-1] * 4
        right_walls = [4] * 4
        bottom_walls = [4] * 4
        for i, j in product(iterator, jterator):
            if self.board[i][j] == 0:
                continue
            # Find destination
            desti = i + di
            destj = j + dj
            while (top_walls[j] < desti and desti < bottom_walls[j]) and (left_walls[i] < destj and destj < right_walls[i])\
                    and self.board[desti][destj] == 0:
                desti += di
                destj += dj
            # Move and merge
            if (desti <= top_walls[j] or bottom_walls[j] <= desti) or (destj <= left_walls[i] or right_walls[i] <= destj)\
                    or self.board[desti][destj] != self.board[i][j]:
                # we do replacement here, not directly assignment. Because there is case that tile did not move
                # If we assign tile to new pos and assign 0 to old pos (also new pos because cell did not move). Tile will be deleted
                t = self.board[desti-di][destj-dj]
                self.board[desti-di][destj-dj] = self.board[i][j]
                self.board[i][j] = t
                modified = modified or t == 0
            else:
                self.board[desti][destj] += self.board[i][j]
                self.board[i][j] = 0
                modified = True
                # Update the wall. so that, merged cell will not merge again
                if desti > i:
                    bottom_walls[j] = desti
                elif desti < i:
                    top_walls[j] = desti
                if destj > j:
                    right_walls[i] = destj
                elif destj < j:
                    left_walls[i] = destj
                # update score
                self.score += self.board[desti][destj]
        # Put new tile after moved
        not modified or self.__put_random_new_tile()

    def restart(self):
        '''Restart the game'''
        self.board = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.__put_random_new_tile()
        self.__put_random_new_tile()
        self.score = 0

    def is_end(self):
        '''Check gameover'''
        for i, j in product(range(4), range(4)):
            tile = self.board[i][j]
            if self.board[i][j] == 0 or \
                (i + 1 < 4 and self.board[i+1][j] == tile) or\
                (j + 1 < 4 and self.board[i][j+1] == tile):
                return False
        return True

    def __put_random_new_tile(self):
        '''
        Randomly created new tile 2
        '''
        choices = [(i, j) for (i, j) in product(range(4), range(4)) if self.board[i][j] == 0]
        if len(choices) == 0:
            return
        r, c = choice(choices)
        self.board[r][c] = 2 if random() > 0.5 else 4
