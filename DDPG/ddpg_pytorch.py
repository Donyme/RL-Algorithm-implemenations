import os
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import Adam
from Replay_buffer import ReplayBuffer
from Networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, input_dims, alpha = 0.001, beta = 0.002, 
                 env = None, gamma = 0.9, n_actions = 2, max_size = 1000000, 
                 tau = 0.05, fc1 = 400, fc2 = 300, batch_size = 64, noise = 0.1):
        self.gamma = gamma
        self.tau   = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.noise = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.batch_size = batch_size

        self.actor = ActorNetwork(n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(name='critic')
        self.target_actor = ActorNetwork(n_actions=n_actions, name='target_actor')
        self.target_critic = CriticNetwork(name='target_critic')

        self.actor_optimizer = Adam(self.actor.parameters(), lr = alpha)
        self.critic_optimizer = Adam(self.critic.parameters(), lr = beta)

        self.update_network_parameters(tau = 1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        self.target_actor.load_state_dict(self.actor.state_dict())
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * (target_param.data))

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * (target_param.data))

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print("....Saving Models....")
        torch.save(self.actor.state_dict, self.actor.checkpoint_file)
        torch.save(self.critic.state_dict, self.critic.checkpoint_file)
        torch.save(self.target_actor.state_dict, self.target_actor.checkpoint_file)
        torch.save(self.target_critic.state_dict, self.target_critic.checkpoint_file)

    def load_models(self):
        print("....Loading Models....")
        self.actor.load_state_dict(self.actor.checkpoint_file)
        self.critic.load_state_dict(self.critic.checkpoint_file)
        self.target_actor.load_state_dict(self.target_actor.checkpoint_file)       
        self.target_critic.load_state_dict(self.target_critic.checkpoint_file)

    def choose_action(self, observation, eval = False):
        state = torch.Tensor([observation])
        actions = self.actor(state)

        if not eval:
            actions += torch.normal(mean = 0.0, std = self.noise, size = list(actions.size()))

        # print("Minimum action: {} Maximum action: {}".format(self.min_action, self.max_action))
        actions = torch.clamp(actions, min = self.min_action, max = self.max_action)

        return actions.detach().numpy()[0]

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = torch.Tensor(state)
        actions = torch.Tensor(action)
        rewards = torch.Tensor(reward)
        new_states = torch.Tensor(new_state)
        dones = torch.Tensor(done)

        target_actions = self.target_actor(new_states)
        new_critic_value = torch.squeeze(self.target_critic(new_states, target_actions), 1)
        critic_value = torch.squeeze(self.critic(states, actions), 1)
        target = rewards + self.gamma * new_critic_value * (1-dones)
        critic_loss = MSELoss()
        loss = critic_loss(target, critic_value)

        loss.backward()
        self.critic_optimizer.step()

        new_policy_actions = self.actor(states)
        actor_loss = -self.critic(states, new_policy_actions) # not actor loss but actually Q-value, so gradient ascent
        actor_loss = torch.mean(actor_loss)

        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_network_parameters()

