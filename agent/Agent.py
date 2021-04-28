import numpy as np
from agent.ReplayBuffer import ReplayBuffer

from agent import tf, keras


MEMORY_LENGTH = 100_000
BATCH_SIZE = 128
UPDATE_ACTION_MODEL = 4
UPDATE_TRAIN_MODEL = 1_000
GAMMA_DECAY = 1e-3
GAMMA_MINIMUM = 1e-3


class Agent:
    def __init__(self, agentId, environment_shape, actions_number):
        self.id = agentId
        self.discount_factor = 0.0
        self.replay_buffer = ReplayBuffer(MEMORY_LENGTH, environment_shape)
        self.previous_state = None
        self.previous_action = None
        self.iteration_counter = 0
        self.actions_number = actions_number
        self.gamma = 0.9
        # Being trained really often
        self.action_model = self.create_model(environment_shape, actions_number)

        # Used to train other model. Being updated rarely
        self.train_model = self.create_model(environment_shape, actions_number)
        self.train_model.set_weights(self.action_model.get_weights())
        self.loss = 0

    @staticmethod
    def create_model(environment_shape, actions_number, learning_rate=0.001):
        '''
        Create and returns the actor model.
            Dense(64)
            Dense(64)
            Dense(actions_number, activation="linear"))

        Args:

        environment_shape: Dimension of data from environment.

        actions_number: Count of available actions.
        '''
        model = keras.models.Sequential([
            keras.layers.Dense(64, activation='tanh', input_shape=environment_shape),
            keras.layers.GaussianDropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(actions_number, activation="linear")
        ])
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=learning_rate))
        return model

    def act(self, environment_state):   
        print(self.gamma)
        self.previous_state = environment_state
        if self.gamma < np.random.rand():
            self.previous_action = tf.math.argmax(self.action_model(environment_state)[0])
        else:
            self.previous_action = np.random.randint(0, self.actions_number)
        return self.previous_action

    def train(self):
        if BATCH_SIZE <= self.replay_buffer.real_length and self.iteration_counter % UPDATE_ACTION_MODEL == 0:
            current_states, future_states, actions, rewards, done_flags = \
                self.replay_buffer.get_experience(BATCH_SIZE)
            current_q_values = self.action_model.predict(current_states)
            future_q_values = self.train_model.predict(future_states)
            target_q_values = np.copy(current_q_values)
            batch_index = np.arange(actions.size, dtype=np.int32)
            target_q_values[batch_index, actions] = \
                rewards + self.discount_factor * np.max(future_q_values, axis=1) * done_flags
            self.loss = self.action_model.train_on_batch(current_states, target_q_values)
            if self.iteration_counter % UPDATE_TRAIN_MODEL == 0:
                self.train_model.set_weights(self.action_model.get_weights())
                if self.discount_factor < 0.9:
                    self.discount_factor += 0.1
        self.iteration_counter += 1
        if self.gamma > GAMMA_MINIMUM:
            self.gamma -= GAMMA_DECAY

    def remember(self, next_state, reward, done_flag):
        self.replay_buffer.remember(self.previous_state, next_state,
                                    self.previous_action, reward, int(not done_flag))


if __name__ == "__main__":
    pass
