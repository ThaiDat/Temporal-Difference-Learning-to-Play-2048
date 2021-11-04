from globalconfig import gconfig
import torch
from torch import nn


class DQN2048Agent(nn.Module):
    '''Pytorch network to learn to play 2048'''
    def __init__(self, epsilon=gconfig['INITIAL_EPSILON']):
        '''
        epsilon: initial value of epsilon greedy policy
        '''
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = 4
        self.qnetwork = nn.Sequential(
            nn.Conv2d(16, 64, 2), # 64x3x3
            nn.ReLU(),
            nn.Conv2d(64, 128, 2), # 128x2x2
            nn.ReLU(),
            nn.Conv2d(128, 512, 2), # 512x1x1
            nn.ReLU(),
            nn.Flatten(), # 512
            nn.Linear(512, 512), # 512
            nn.ReLU(),
            nn.Linear(512, 256), # 256
            nn.ReLU(),
            nn.Linear(256, 128), # 128
            nn.ReLU(),
            nn.Linear(128, 64), # 64
            nn.ReLU(),
            nn.Linear(64, self.n_actions) # 4
        )

    def forward(self, states):
        '''
        Forward the states tensor to receive tensor of qvalues
        states: tensor of states batch x 16x4x4
        return tensor of qvalues batch x n_actions
        '''
        # Use your network to compute qvalues for given state
        qvalues = self.qnetwork(states) # <YOUR CODE>
        return qvalues

    def sample_actions(self, qvalues):
        '''
        Pick actions given qvalues. Uses epsilon-greedy exploration strategy.
        qvalues: Tensor of qvalues batch x n_actions
        return tensor of actions batch
        '''
        max_indices = torch.argmax(qvalues, dim=-1)
        minor_prob = self.epsilon / self.n_actions
        major_prob = 1 - self.epsilon + minor_prob
        probs = torch.full(qvalues.size(), minor_prob)
        probs[torch.arange(qvalues.size(0)), max_indices] = major_prob
        return torch.multinomial(probs, 1)