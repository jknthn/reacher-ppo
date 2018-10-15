import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FullyConnectedNetwork(nn.Module):
    
    def __init__(self, state_size, output_size, hidden_size, output_gate=None):
        super(FullyConnectedNetwork, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.output_gate = output_gate

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        if self.output_gate:
            x = self.output_gate(x)
        return x


class PPOPolicyNetwork(nn.Module):
    
    def __init__(self, config):
        super(PPOPolicyNetwork, self).__init__()
        state_size = config['environment']['state_size']
        action_size = config['environment']['action_size']
        hidden_size = config['hyperparameters']['hidden_size']
        device = config['pytorch']['device']

        self.actor_body = FullyConnectedNetwork(state_size, action_size, hidden_size, F.tanh)
        self.critic_body = FullyConnectedNetwork(state_size, 1, hidden_size)  
        self.std = nn.Parameter(torch.ones(1, action_size))
        self.to(device)

    def forward(self, obs, action=None):
        obs = torch.Tensor(obs)
        a = self.actor_body(obs)
        v = self.critic_body(obs)
        
        dist = torch.distributions.Normal(a, self.std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return action, log_prob, torch.Tensor(np.zeros((log_prob.size(0), 1))), v