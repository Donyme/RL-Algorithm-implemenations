import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical

from Networks import ActorCriticNetwork

class Agent:
    def __init__(self, alpha=0.0001, gamma = 0.99, n_actions = 2):
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]

        self.actor_critic  = ActorCriticNetwork(n_actions=self.n_actions)
        self.optimizer = Adam(self.actor_critic.parameters(), lr = alpha)

    def choose_action(self, observation):
        state = torch.Tensor([observation])
        _, probs = self.actor_critic(state)

        action_probs = Categorical(probs=probs)
        action = action_probs.sample()
        self.action = action

        return self.action.numpy()[0]
    
    def save_models(self):
        print("...Saving models...")
        torch.save(self.actor_critic.state_dict, self.actor_critic.checkpoint_file)

    def load_models(self):
        print("..Loading models...")
        self.actor_critic.load_state_dict(self.actor_critic.checkpoint_file)

    def learn(self, state, reward, state_, done):
        state  = torch.Tensor([state])
        state_ = torch.Tensor([state_])
        reward = torch.Tensor([reward], dtype = torch.float32)

        state_value,  probs = self.actor_critic(state)
        state_value_, _ = self.actor_critic(state_)
        state_value     = torch.squeeze(state_value)
        state_value_    = torch.squeeze(state_value_)

        action_probs = Categorical(probs=probs)
        log_prob = action_probs.log_prob(self.action)

        delta = reward + self.gamma * state_value_*(1-int(done) - state_value)
        actor_loss = -log_prob * delta
        critic_loss = delta**2

        total_loss = actor_loss + critic_loss

        total_loss.backward()
        self.optimizer.step()