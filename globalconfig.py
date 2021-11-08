from os import path


gconfig = dict()

# Browser to play game
# Require necessary driver to run successfully. In this project, geckodriver v0.30.0x64 is included by default.
# If you want to switch browsers or run on another system, download the proper driver.
gconfig['BROWSER'] = 'Firefox'

# Path to browser driver file
# Help this program to find and use the driver
gconfig['BROWSER_DRIVER_FILE'] = path.join('drivers', 'geckodriver.exe')

# URL of the web game used by the environment
gconfig['GAME_URL'] = 'https://2048game.com'

# Map between numbers and actions
gconfig['ACTION_MAP'] = {0:'LEFT', 1:'UP', 2:'RIGHT', 3:'DOWN'}

# Sleep time after sending action to the game
# This is for avoiding error when you play too fast
gconfig['ACTION_SLEEP'] = 0.1

# Scale the reward for the q-value approximator not to learn too big values
gconfig['REWARD_SCALE'] = 1/128

# Number of environments / batch size in A3C
gconfig['BATCH'] = 4

# Gradient descent optimizer. Must be exact the same with torch optim class name
gconfig['OPTIMIZER'] = 'Adam'

# Learning rate of model
gconfig['LEARNING_RATE'] = 1e-2

# Clip the gradient norm to this number.
gconfig['MAX_GRADIENT_NORM'] = 100

# Length of one-hot encoded cell. Should be >= 11 as 2048->10
gconfig['CHANEL_ENCODED'] = 16

# Device to train network on. cuda or cpu
gconfig['DEVICE'] = 'cpu'

# Discounted factor gamma in Bellman (optimality) equation. How reward in later states affect the current one
gconfig['DISCOUNTED'] = 0.8

# We will just monitor some number of last steps. For example -500 means that only last 500 steps will display on the screen
gconfig['MONITOR_RANGE'] = -1000

# Total number of training steps
gconfig['TRAIN_STEPS'] = 1_000_000

# We will updates some information like gradient and loss after some steps to help monitor the training process
gconfig['MONITOR_STEPS'] = 1000

# Save the models to file every n steps
gconfig['BACKUP_STEPS'] = 10_000

# Evaluate the agent every n steps
gconfig['EVALUATE_STEPS'] = 1000

# The location to save model
gconfig['BACKUP_LOCATION'] = path.join('bin', 'model.rl')

# Define patterns of weightless network
gconfig['NTUPLES'] = [
    [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1)],
    [(1,0), (1,1), (1,2), (1,3), (2,0), (2,1)],
    [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)],
    [(1,0), (1,1), (1,2), (2,0), (2,1), (2,2)],
]