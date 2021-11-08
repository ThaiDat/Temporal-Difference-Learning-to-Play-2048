from globalconfig import gconfig
from gameenv import GameEnv, EnvironementBatch, encode_board
from gamedriver import SilentGameDriver2048
from gameplanner import plan
from agent import NeuralNetworkModel
import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def evaluate(model, n=1, device=gconfig['DEVICE']):
    '''
    Evaluate model on practical environment
    model: Model to evaluate
    n: Number of games
    device: device of model
    return mean of max tile, meean of score
    '''
    env = GameEnv(SilentGameDriver2048())
    mean_max_tile = 0
    mean_score = 0
    for i in range(n):
        s = env.reset()
        done = False
        while not done:
            a, v = plan([s], model)
            sp, r, done = env.step(a[0])
            s = sp
        max_tile = np.max(env.driver.get_board())
        score = env.driver.get_score()
        mean_max_tile = (mean_max_tile * i + max_tile) / (i + 1)
        mean_score = (mean_score * i + score) / (i + 1)
    return mean_max_tile, mean_score
            

def prepare_batch(states, values, device=gconfig['DEVICE']):
    '''
    Prepare batch for training the network
    states: List of pure board representation
    values: list of state values
    return states tensor, values tensor
    '''
    states = np.array([encode_board(s) for s in states], dtype=np.float32)
    generated_states = [states]
    for i in range(3):
        generated_states.append(np.rot90(generated_states[-1], 1, (2, 3)))
    states = np.concatenate(generated_states)
    values = np.tile(values, 4).reshape(-1, 1)
    return torch.tensor(states, device=device, dtype=torch.float), torch.tensor(values, device=device, dtype=torch.float)


if __name__=='__main__':
    # Constants
    gamma = gconfig['DISCOUNTED']
    device = gconfig['DEVICE']

    # Init elements for learning
    envs = EnvironementBatch(1)
    model = NeuralNetworkModel()

    # Logging
    history_loss = []
    history_grad = []
    history_maxtile = []
    history_score = []
    plt.ion()
    fig, axs = plt.subplots(2, 2)
    axs[0][0].set_title('Loss')
    axs[1][0].set_title('Gradient')
    axs[1][0].set_title('Max tile')
    axs[1][1].set_title('Score')

    states = envs.reset()
    for i in tqdm(range(gconfig['TRAIN_STEPS'])):
        # Play
        actions, state_values = plan(states, model, gamma=gamma, device=device)
        nxt_states, rewards, done = envs.step(actions)
        # Improving. We ignore rewards and done here since we obtained enough information for training in planning step
        states_batch, values_batch = prepare_batch(states, state_values)
        loss, grad = model.fit(states_batch, values_batch)
        
        # Next
        states = nxt_states

        # Logging
        history_loss.append(loss)
        history_grad.append(grad)
        if i % gconfig['EVALUATE_STEPS'] == 0:
            mean_max_tile, mean_score = evaluate(model, n=5)
            history_maxtile.append(mean_max_tile)
            history_score.append(mean_score)

        # Monitor
        if i % gconfig['MONITOR_STEPS'] == 0:
            axs[0][0].cla()
            axs[0][0].plot(history_loss)
            axs[0][0].set_title('Loss')
            axs[1][0].cla()
            axs[1][0].plot(history_grad)
            axs[1][0].set_title('Gradient')
            axs[0][1].cla()
            axs[0][1].plot(history_maxtile)
            axs[0][1].set_title('Max tile')
            axs[1][1].cla()
            axs[1][1].plot(history_score)
            axs[1][1].set_title('Score')
            fig.canvas.draw()
            fig.canvas.flush_events()           

        # Back-up
        if i % gconfig['BACKUP_STEPS'] == 0:
            torch.save(model.network.state_dict(), gconfig['BACKUP_LOCATION'])
