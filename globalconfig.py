from os import path


gconfig = dict()

# Browser to play game
# Require necessary driver to run successfully. In this project, geckodriver v0.30.0x64 is included by default.
# If you want to switch browsers or run on another system, download the proper driver.
gconfig['BROWSER'] = 'Firefox'

# Path to browser driver file
# Help this program to find and use the driver
gconfig['BROWSER_DRIVER_FILE'] = path.join('drivers', 'geckodriver.exe')

# Position of browser when open
gconfig['BROWSER_POSITION'] = (0, 0)

# Size of browser when open
gconfig['BROWSER_SIZE'] = (500, 750)

# URL of the web game used by the environment
gconfig['GAME_URL'] = 'https://2048game.com'

# Map between numbers and actions
gconfig['ACTION_MAP'] = {0:'LEFT', 1:'UP', 2:'RIGHT', 3:'DOWN'}

# Sleep time after sending action to the game
# This is for avoiding error when you paly too fast
gconfig['ACTION_SLEEP'] = 0.05