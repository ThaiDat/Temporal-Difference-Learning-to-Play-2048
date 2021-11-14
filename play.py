from globalconfig import gconfig
from gameenv import GameEnv
from gamedriver import WebGameDriver2048
from agent import WeightlessNetworkModel
from gameplanner import plan
from pickle import load


if __name__=='__main__':
    model = None
    with open(gconfig['TEST_MODEL_FILE'], 'rb') as f:
        model = load(f)
    
    env = GameEnv(WebGameDriver2048())
    state = env.reset()
    done = False
    while not done:
        action, value = plan([state], model)
        nxtstate, reward, done = env.step(action[0])
        state = nxtstate
