from globalconfig import gconfig
from gameenv import GameEnv, EnvironementBatch, encode_board
from gamedriver import SilentGameDriver2048
from gameplanner import plan
from agent import WeightlessNetworkModel
import numpy as np
import time
from matplotlib import pyplot as plt
from collections import Counter
from pickle import dump


def evaluate(model, n=gconfig['EVALUATE_GAMES'], device=gconfig['DEVICE']):
    '''
    Evaluate model on practical environment
    model: Model to evaluate
    n: Number of games
    device: device of model
    return list of max tiles, list of scores
    '''
    env = GameEnv(SilentGameDriver2048())
    max_tiles = []
    scores = []
    for i in range(n):
        s = env.reset()
        done = False
        while not done:
            a, v = plan([s], model)
            sp, r, done = env.step(a[0])
            s = sp
        max_tile = np.max(env.driver.get_board())
        max_tiles.append(max_tile)
        scores.append(env.driver.get_score())
    return max_tiles, scores
            

if __name__=='__main__':
    # Constants
    gamma = gconfig['DISCOUNTED']

    # Init elements for learning
    envs = EnvironementBatch(1)
    model = WeightlessNetworkModel()

    states = envs.reset()
    alltime_max = 0
    checkpoint = time.process_time()
    for i in range(gconfig['TRAIN_STEPS']):
        # Play
        actions, state_values = plan(states, model, gamma=gamma)
        nxt_states, rewards, done = envs.step(actions)
        # Improving. We ignore rewards and done here since we obtained enough information for training in planning step
        loss = model.fit(states, state_values)     
        # Next
        alltime_max = max(alltime_max, np.max(states))
        states = nxt_states
        # Eval
        print(i, end='\r')
        if i % gconfig['EVALUATE_STEPS'] == 0:
            # Train Checkpoint
            new_checkpoint = time.process_time()
            elapsed = new_checkpoint - checkpoint
            checkpoint = new_checkpoint
            # evaluate 
            n_games = gconfig['EVALUATE_GAMES']
            max_tiles, scores = evaluate(model, n=n_games)
            score_min = min(scores)
            score_max = max(scores)
            score_mean = np.mean(scores)
            tile_counter = Counter(max_tiles)
            # Evaluate checkpoint
            new_checkpoint = time.process_time()
            eval_elapsed = new_checkpoint - checkpoint
            checkpoint = new_checkpoint
            # Logging
            print('Trained {n_steps} steps in {train_time} seconds'.format(n_steps=i, train_time=elapsed))
            print('    Maximum tile reached while training: {all_max}'.format(all_max=alltime_max))
            print('    Evaluate {n_games} games in {eval_time} seconds'.format(n_games=n_games, eval_time=eval_elapsed))
            print('    Score - Min:{0}, Max:{1}, Mean:{2:.2f}'.format(score_min, score_max, score_mean))
            print('    Max tiles - ', end='')
            tile = min(tile_counter)
            tile_max = max(tile_counter)
            while tile <= tile_max:
                print('{tile}:{percentage:.1f}%'.format(tile=tile, percentage=tile_counter[tile]/n_games*100), end=' , ')
                tile *= 2
            print('')            
        # Back-up
        if i % gconfig['BACKUP_STEPS'] == 0:
            with open(gconfig['BACKUP_LOCATION'], 'wb') as f:
                dump(model, f)
