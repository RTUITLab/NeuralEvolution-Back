import numpy as np


class ReplayBuffer:

    def __init__(self, length, environment_shape):
        self.pointer = 0
        self.real_length = 0
        self.max_length = length
        self.current_states = np.empty((length, *environment_shape), dtype=np.float32)
        self.future_states = np.empty((length, *environment_shape), dtype=np.float32)
        self.actions = np.empty(length, dtype=np.int32)
        self.rewards = np.empty(length, dtype=np.float32)
        self.done_flags = np.empty(length, dtype=np.bool)
        pass

    def remember(self, current_state, future_state, action, reward, done_flag):
        self.current_states[self.pointer] = current_state
        self.future_states[self.pointer] = future_state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.done_flags[self.pointer] = done_flag
        self.real_length += int(self.real_length < self.max_length)
        self.pointer = (self.pointer + 1) % self.max_length

    def get_experience(self, batch_size):
        batch = np.random.choice(self.real_length, batch_size, replace=False)
        current_states = self.current_states[batch]
        future_states = self.future_states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        done_flags = self.done_flags[batch]
        return current_states, future_states, actions, rewards, done_flags


if __name__ == "__main__":
    pass