
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from Networks import MLPModel

class Agent:
    def __init__(self, alpha=0.0001, n_actions = 2):
        self.n_actions = n_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]

        self.MLPModel  = MLPModel(n_actions = self.n_actions)
        self.optimizer = Adam(self.MLPModel.parameters(), lr = alpha)

    def choose_action(self, observation):
        state = torch.Tensor([observation])
        probs = self.MLPModel(state)

        action_probs = Categorical(probs=probs) # distribution of action space
        action = action_probs.sample() # action sampling
        self.action = action

        return self.action.numpy()[0]
    
    def save_models(self):
        print("...Saving models...")
        torch.save(self.MLPModel.state_dict, self.MLPModel.checkpoint_file)

    def load_models(self):
        print("..Loading models...")
        self.MLPModel.load_state_dict(self.MLPModel.checkpoint_file)

    def learn(self, batch_states, batch_actions, batch_weights):
        state  = torch.Tensor([batch_states])
        action = torch.Tensor([batch_actions])
        reward = torch.Tensor([batch_weights])

        probs = self.MLPModel(state)
        action_probs = Categorical(probs=probs)
        logp = action_probs.log_prob(action)
        loss = -(logp * reward).mean()

        loss.backward()
        self.optimizer.step()