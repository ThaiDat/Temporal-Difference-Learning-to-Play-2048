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

# Position of training metrics monitor on the screen
gconfig['TRAIN_FIGURE_POSITION'] = (501, 0)

# Position of game metrics monitor on the screen
gconfig['GAME_FIGURE_POSITION'] = (1002, 0)

# URL of the web game used by the environment
gconfig['GAME_URL'] = 'https://2048game.com'

# Map between numbers and actions
gconfig['ACTION_MAP'] = {0:'LEFT', 1:'UP', 2:'RIGHT', 3:'DOWN'}

# Sleep time after sending action to the game
# This is for avoiding error when you paly too fast
gconfig['ACTION_SLEEP'] = 0.02

# Scale the reward for the q-value approximator not to learn too big values
gconfig['REWARD_SCALE'] = 1/2048

# Size of experiences replay
gconfig['EXPERIENCE_BUFFER'] = 2048

# Minimum of experiences to sample from
# If buffer has less experience than this number, sample nothing
gconfig['MIN_EXPERIENCE'] = 1024

# Size of batch sample from experiences buffer to train
gconfig['BATCH'] = 32

# Gradient descent optimizer. Must be exact the same with torch optim class name
gconfig['OPTIMIZER'] = 'Adam'

# Learning rate of model
gconfig['LEARNING_RATE'] = 1e-4

# Clip the gradient norm to this number.
gconfig['MAX_GRADIENT_NORM'] = 50

# Length of one-hot encoded cell. Should be >= 11 as 2048->10
gconfig['CHANEL_ENCODED'] = 16

# Device to train network on. cuda or cpu
gconfig['DEVICE'] = 'cuda'

# Discounted factor gamma in Bellman (optimality) equation. How reward in later states affect the current one
gconfig['DISCOUNTED'] = 0.99

# Probability for choosing random actions while training
gconfig['INITIAL_EPSILON'] = 1

# Minimum probability for choosing random actions while training. This is for keeping exploring to find better policy
gconfig['MIN_EPSILON'] = 0.1

# The epsilon decrement every steps until reach minimum value
# Epsilon need decay overtime (less exploration, more exploitation). 
gconfig['EPSILON_LINEAR_DECAY'] = (1 - 0.1) / 7000

# Total number of training steps
gconfig['TRAIN_STEPS'] = 5000

# After some steps, we will update target network parameter using learning network/agent
gconfig['UPDATE_STEPS'] = 500

# We will updates some information like gradient and loss after some steps to help monitor the training process
gconfig['LOG_STEPS'] = 10