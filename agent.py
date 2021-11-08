from globalconfig import gconfig
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import utils


class NeuralNetwork(nn.Module):
    '''Pytorch network to learn values function'''
    def __init__(self):
        '''
        epsilon: initial value of epsilon greedy policy
        '''
        super(NeuralNetwork, self).__init__()
        self.hoz_conv = nn.Sequential(
            nn.Conv2d(16, 64, (2,1)), # 64x3x4
            nn.ReLU(),
            nn.Conv2d(64, 256, (2, 1)), # 256x2x4
            nn.ReLU(),
            nn.Conv2d(256, 1024, (2, 1)), # 1024x1x4
            nn.ReLU(),
            nn.Flatten(), # 4096
        )
        self.ver_conv = nn.Sequential(
            nn.Conv2d(16, 64, (1, 2)), # 64x4x3
            nn.ReLU(),
            nn.Conv2d(64, 256, (1, 2)), # 256x4x2
            nn.ReLU(),
            nn.Conv2d(256, 1024, (1, 2)), # 1024x4x1
            nn.ReLU(),
            nn.Flatten(), # 4096
        )
        self.box_conv = nn.Sequential(
            nn.Conv2d(16, 64, 2), # 64x3x3
            nn.ReLU(),
            nn.Conv2d(64, 256, 2), # 256x2x2
            nn.ReLU(),
            nn.Conv2d(256, 1024, 2), # 1024x1x1
            nn.ReLU(),
            nn.Flatten(), # 1024
        )
        self.v = nn.Sequential(
            nn.Linear(4096+4096+1024, 1024), # 1024
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, states):
        '''
        Forward the states tensor to receive tensor of qvalues
        states: tensor of states batch x 16x4x4
        return logit and state values
        '''
        hoz = self.hoz_conv(states) 
        ver = self.ver_conv(states)
        box = self.box_conv(states)
        fs = torch.cat((hoz, ver, box), -1)
        v = self.v(fs)
        return v


class NeuralNetworkModel:
    '''
    Model of value function. Wrapper to train the network
    '''
    def __init__(self, make_network=NeuralNetwork, device=gconfig['DEVICE'], optim=gconfig['OPTIMIZER'], lr=gconfig['LEARNING_RATE']):
        '''
        make_network: function or class to init network
        optim: Name of optimizer
        lr: Learning rate
        '''
        self.network = make_network().to(device)
        self.optim = getattr(torch.optim, optim)(self.network.parameters(), lr=lr)

    def predict(self, states):
        '''
        Predict values of given states
        states: tensor of encoded states. batch x16x4x4
        return state values batch
        '''
        with torch.no_grad():
            return self.network(states)
    
    def fit(self, states, values, max_grad_norm=gconfig['MAX_GRADIENT_NORM']):
        '''
        Train the network with given datas
        states: tensor of encoded states. batch x16x4x4
        values: tensor of state values. batch
        return loss, gradient values
        '''
        self.optim.zero_grad()
        pr = self.network(states)
        loss = F.smooth_l1_loss(pr, values)
        loss.backward()
        grad_norm = utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
        self.optim.step()
        return loss.cpu().item(), grad_norm.cpu().item()