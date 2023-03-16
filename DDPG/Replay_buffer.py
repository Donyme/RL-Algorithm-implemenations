import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_counter      = 0
        self.state_memory     = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory    = np.zeros((self.mem_size, n_actions))
        self.reward_memory    = np.zeros(self.mem_size)
        self.terminal_memory  = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state, done):
        idx = self.mem_counter % self.mem_size

        self.state_memory[idx]     = state
        self.new_state_memory[idx] = new_state
        self.action_memory[idx]    = action
        self.reward_memory[idx]    = reward
        self.terminal_memory[idx]  = done

        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_memory = min(self.mem_counter, self.mem_size)

        batch = np.random.choice(max_memory, batch_size, replace=False)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.action_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones