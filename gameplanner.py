from globalconfig import gconfig
from gamedriver import get_after_state, check_terminal_board
from itertools import product
import numpy as np


def get_possible_new_states(state, terminal_included=False):
    '''
    Get all possible states by putting a random tile in this current state
    state: pure board representation
    terminal_included: If True, all possible next states return. If false, terminal state will be ignored
    return list of new states, probability of new states
    '''
    new_states = []
    probs = []
    sum_prob = 0
    for i, j in product(range(4), range(4)):
        if state[i][j] == 0:
            # add 2
            new_board = [row[:] for row in state]
            new_board[i][j] = 2
            if terminal_included or not check_terminal_board(new_board):
                new_states.append(new_board)
                probs.append(0.9)
            # add 4
            new_board = [row[:] for row in state]
            new_board[i][j] = 4
            if terminal_included or not check_terminal_board(new_board):
                new_states.append(new_board)
                probs.append(0.1)
            sum_prob += 1
    probs = np.array(probs) / sum_prob
    return new_states, probs


def plan(states, vf, gamma=gconfig['DISCOUNTED']):
    '''
    Plan and select the best move with the help of network
    states: batch of pure board representation
    vf: state value function. Should be a Model (see agent.py)
    gamma: discounted factor how next state affect current states.
    return batch of actions corresponding to states, batch value function corresponding to state (which is equal to action value function of optimal action)
    '''
    actions = []
    values = []
    for state in states:
        best_a = None
        best_v = float('-inf')
        # Try all actions
        for a in range(4):
            afterstate = [row[:] for row in state]
            score, modified = get_after_state(afterstate, a)            
            if not modified:
                continue
            # reward = int(np.max(afterstate) > max_tile)
            reward = score * gconfig['REWARD_SCALE']
            new_states, probs = get_possible_new_states(afterstate)
            next_v = 0
            if len(new_states) > 0:
                next_v = vf.predict(new_states)
                next_v = (probs @ next_v).item()
            v = reward + gamma * next_v
            if v >= best_v:
                best_v = v
                best_a = a
        actions.append(best_a)
        values.append(best_v)
    return actions, values