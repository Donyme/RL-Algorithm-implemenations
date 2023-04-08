import os
import torch
import torch.nn as nn

class ActorCriticNetwork(nn.Module):
    def __init__(self, n_actions, fc1_dims = 1024, fc2_dims = 512,
                name='actor_critic', chkpt_dir = "tmp/actor_critic"):
        super(ActorCriticNetwork, self).__init__()
        self.fc1_dims        = fc1_dims
        self.fc2_dims        = fc2_dims
        self.n_actions       = n_actions
        self.model_name      = name
        self.checkpoint_dir  = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ac')
        
        self.fc1   = nn.LazyLinear(out_features = self.fc1_dims)
        self.fc2   = nn.LazyLinear(out_features = self.fc2_dims)
        self.relu  = nn.ReLU()
        self.value = nn.LazyLinear(out_features = 1)
        self.pi    = nn.LazyLinear(out_features = self.n_actions)
        self.softmax = nn.Softmax() 

    def forward(self, state):
        val = self.fc1(state)
        val = self.fc2(val)

        v   = self.value(val)
        pi  = self.pi(val)
        pi  = self.softmax(pi)

        return v, pi 
        