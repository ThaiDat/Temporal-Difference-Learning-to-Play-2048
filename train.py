from globalconfig import gconfig
from gamedriver import WebGameDriver2048
from gameenv import GameEnv
from experience import ExperienceReplay
from agent import DQN2048Agent
from matplotlib import pyplot as plt
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

    # Monitoring
    plt.ion()
    train_fig, train_axs = plt.subplots(2, 1, sharex=True, figsize=(5, 6))
    train_axs[0].set_title('Gradient Norm')
    train_axs[1].set_title('TD Loss')
    train_fig.suptitle('Epsion = %f'%gconfig['INITIAL_EPSILON'])
    train_grads = []; train_losses = []

    game_fig, game_axs = plt.subplots(2, 1, sharex=True, figsize=(5, 6))
    game_axs[0].set_title('Max rewards per game')
    game_axs[1].set_title('Scores per game')
    game_scores = []; game_max_rewards = []

    # This position is my screen specific
    train_fig.canvas.manager.window.move(*gconfig['TRAIN_FIGURE_POSITION'])
    game_fig.canvas.manager.window.move(*gconfig['GAME_FIGURE_POSITION'])
    plt.show()

    # init elements for training
    env = GameEnv(WebGameDriver2048())
    agent = DQN2048Agent(gconfig['INITIAL_EPSILON']).to(device)
    target_network = DQN2048Agent().to(device)
    target_network.load_state_dict(agent.state_dict())
    memory = ExperienceReplay()
    optimizer = getattr(torch.optim, gconfig['OPTIMIZER'])(agent.parameters(), lr=gconfig['LEARNING_RATE'])

    # train loop
    gc.collect()
    episode_reward = 0
    s = env.reset()
    for i in range(gconfig['TRAIN_STEPS']):
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
        episode_reward += r
        max_tile = np.max(env.driver.get_board())

        # next step
        agent.epsilon = max(agent.epsilon - epsiolon_lin_decay, min_epsilon)
        s = env.reset() if done else sp

        # Update network
        if i % gconfig['UPDATE_STEPS'] == 0:
            target_network.load_state_dict(agent.state_dict())
        
        # Monitoring
        if done:
            game_scores.append(episode_reward)
            game_max_rewards.append(max_tile)
            episode_reward = 0
            # Monitor after an game end
            game_axs[0].cla()
            game_axs[0].plot(game_max_rewards)
            game_axs[0].set_title('Max rewards per game')
            game_axs[1].cla()
            game_axs[1].plot(game_scores)
            game_axs[1].set_title('Scores per game')
            game_fig.canvas.draw()
            game_fig.canvas.flush_events()

        if i % gconfig['LOG_STEPS'] == 0:
            train_axs[0].cla()
            train_axs[0].plot(train_grads)
            train_axs[0].set_title('Gradient Norm')
            train_axs[1].cla()
            train_axs[1].plot(train_losses)
            train_axs[1].set_title('TD Loss')
            train_fig.suptitle('Epsion = %f'%agent.epsilon)
            train_fig.canvas.draw()
            train_fig.canvas.flush_events()
