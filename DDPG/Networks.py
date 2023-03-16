import os
import torch
import torch.nn as nn
import sys
sys.path.append("../")

class CriticNetwork(nn.Module):
    def __init__(self, fc1_dims = 512, fc2_dims = 512, name = 'critic', chkptdir = 'results/ddpg'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.checkpoint_dir = chkptdir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.pt')

        self.fc1  = nn.LazyLinear(out_features = self.fc1_dims)
        self.fc2  = nn.LazyLinear(out_features = self.fc2_dims)
        self.relu = nn.ReLU()
        self.q    = nn.LazyLinear(out_features = 1)

    def forward(self, state, action):
        action_value = self.fc1(torch.cat([state, action], dim = 1))
        action_value = self.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = self.relu(action_value)

        q_value = self.q(action_value)

        return q_value
        
class ActorNetwork(nn.Module):
    def __init__(self, fc1_dims = 512, fc2_dims = 512, n_actions = 2, name = 'actor', chkptdir = 'results/ddpg'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims  = fc1_dims
        self.fc2_dims  = fc2_dims
        self.n_actions = n_actions

        self.model_name      = name
        self.checkpoint_dir  = chkptdir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.pt')

        self.fc1  = nn.LazyLinear(out_features = self.fc1_dims)
        self.fc2  = nn.LazyLinear(out_features = self.fc2_dims)
        self.mu   = nn.LazyLinear(out_features = n_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        ac_val = self.fc1(state)
        ac_val = self.relu(ac_val)
        ac_val = self.fc2(state)
        ac_val = self.relu(ac_val)

        mu = self.mu(ac_val)
        mu = self.tanh(mu)

        return mu