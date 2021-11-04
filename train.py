from globalconfig import gconfig
from gamedriver import WebGameDriver2048, SilentGameDriver2048
from gameenv import GameEnv
from experience import ExperienceReplay
from agent import DQN2048Agent
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import gc
import torch
from torch.nn import utils

def compute_td_loss(states, actions, rewards, nxt_states, done, network, target_network,
                    gamma=gconfig['DISCOUNTED'], device=gconfig['DEVICE']):
    '''
    Compute temporal difference loss. Use backpropagation from this loss to improve agent
    states: list of states
    actions: list of actions the agent did in that states
    rewards: list of rewards the agent received by doing these actions
    nxt_states: list of states the agents observed after doing these actions
    done: true if nxt_states is terminal states
    network: Network to train. Or the agent
    target_network: Network to learn from. Qvalues approximator
    gamma: Reward discounted factors
    device: Device to train the network on
    return loss with grad
    '''
    # transform inputs to tensors
    # Convert states to numpy first because torch is extremely slow when converting list of ndarray to tensor
    states = torch.tensor(np.array(states), device=device, dtype=torch.float)
    # We just use action as a filter/indices. So, do not need to transform to tensor
    # actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)
    nxt_states = torch.tensor(np.array(nxt_states), device=device, dtype=torch.float)
    done = torch.tensor(done, device=device, dtype=torch.float)
    not_done = 1 - done
    # A values from current states.
    action_values = network(states)
    selected_action_values = action_values[range(len(actions)), actions]
    # Next states values. We will compute it in double q network style.
    # TODO: use double q network
    nxt_action_values = target_network(nxt_states)
    selected_nxt_action_values = torch.amax(nxt_action_values)
    # Next states values. Remember that if done, target action values is reward only since there is no next states
    target_action_values = rewards + gamma * selected_nxt_action_values * not_done
    # MSE loss
    loss = torch.mean(torch.square(target_action_values - selected_action_values))
    return loss


if __name__ == '__main__':
    # constant
    device = gconfig['DEVICE']
    batch_size = gconfig['BATCH']
    min_epsilon = gconfig['MIN_EPSILON']
    epsiolon_lin_decay = gconfig['EPSILON_LINEAR_DECAY']
    max_grad_norm = gconfig['MAX_GRADIENT_NORM']
    monitors = gconfig['MONITOR_RANGE']

    # Monitoring
    plt.ion()
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs[0][0].set_title('Gradient Norm')
    axs[1][0].set_title('TD Loss')
    axs[0][1].set_title('Max rewards per game')
    axs[1][1].set_title('Scores per game')
    game_scores = []; game_max_rewards = []
    train_grads = []; train_losses = []
    fig.suptitle('Epsion = %f'%gconfig['INITIAL_EPSILON'])
    plt.show()

    # init elements for training
    env = GameEnv(SilentGameDriver2048())
    agent = DQN2048Agent(gconfig['INITIAL_EPSILON']).to(device)
    target_network = DQN2048Agent().to(device)
    target_network.load_state_dict(agent.state_dict())
    memory = ExperienceReplay()
    memory.fill(env)
    optimizer = getattr(torch.optim, gconfig['OPTIMIZER'])(agent.parameters(), lr=gconfig['LEARNING_RATE'])

    # train loop
    gc.collect()
    s = env.reset()
    for i in tqdm(range(gconfig['TRAIN_STEPS'])):
        # Play
        st = torch.tensor(s, device=device, dtype=torch.float).unsqueeze(0)
        a = None
        with torch.no_grad():
            qvalues = agent(st)
            a = agent.sample_actions(qvalues).item()
        sp, r, done = env.step(a)
        memory.push(s, a, r, sp, done)
        # Learn
        exp = memory.sample(batch_size)
        if exp is not None:
            states, actions, rewards, nxt_states, dones = exp
            loss = compute_td_loss(states, actions, rewards, nxt_states, dones, agent, target_network)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()
            # Logging
            train_grads.append(grad_norm.data.cpu().item())
            train_losses.append(loss.data.cpu().item())
        
        # Logging
        max_tile = np.max(env.driver.get_board())

        # Update network
        if i % gconfig['UPDATE_STEPS'] == 0:
            target_network.load_state_dict(agent.state_dict())
        
        # Monitoring
        if done:
            game_scores.append(env.driver.get_score())
            game_max_rewards.append(max_tile)
            # Monitor after an game end
            axs[0][1].cla()
            axs[0][1].plot(game_max_rewards[monitors:])
            axs[0][1].set_title('Max rewards per game')
            axs[1][1].cla()
            axs[1][1].plot(game_scores[monitors:])
            axs[1][1].set_title('Scores per game')
            fig.canvas.draw()
            fig.canvas.flush_events()

        if i % gconfig['MONITOR_STEPS'] == 0:
            axs[0][0].cla()
            axs[0][0].plot(train_grads[monitors:])
            axs[0][0].set_title('Gradient Norm')
            axs[1][0].cla()
            axs[1][0].plot(train_losses[monitors:])
            axs[1][0].set_title('TD Loss')
            fig.suptitle('Epsion = %f'%agent.epsilon)
            fig.canvas.draw()
            fig.canvas.flush_events()

        # Backup
        if i % gconfig['BACKUP_STEPS']:
            torch.save(target_network.state_dict(), gconfig['BACKUP_LOCATION'])

        # next step
        agent.epsilon = max(agent.epsilon - epsiolon_lin_decay, min_epsilon)
        s = env.reset() if done else sp

    # Save the model after done
    torch.save(target_network.state_dict(), gconfig['BACKUP_LOCATION'])
    print('DONE')
    plt.ioff()