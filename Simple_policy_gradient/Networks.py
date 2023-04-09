import os
import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, n_actions, fc1_dims = 64, fc2_dims = 32, model_name = "MLP_model", chkpt_dir = "checkpoints/"):
        super(MLPModel, self).__init__()
        self.fc1_dims   = fc1_dims
        self.fc2_dims   = fc2_dims
        self.n_actions  = n_actions
        self.model_name = model_name
        self.chkpt_dir  = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, model_name+'_ac')
        
        self.fc1  = nn.LazyLinear(out_features = self.fc1_dims)
        self.fc2  = nn.LazyLinear(out_features = self.fc2_dims)
        self.relu = nn.ReLU()
        self.pi   = nn.LazyLinear(out_features = self.n_actions)
        self.softmax = nn.Softmax(dim = -1)
        
    def forward(self, observation):
        out = self.fc1(observation)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        pi  = self.pi(out)
        pi  = self.softmax(pi)
        
        return pi