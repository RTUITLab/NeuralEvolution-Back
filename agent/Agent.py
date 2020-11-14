import numpy as np
from agent.ReplayBuffer import ReplayBuffer

from agent import tf, keras


MEMORY_LENGTH = 10_000
BATCH_SIZE = 32
UPDATE_ACTION_MODEL = 4
UPDATE_TRAIN_MODEL = 1_000
GAMMA_DECAY = 1e-4
GAMMA_MINIMUM = 1e-3
DISCOUNT_FACTOR = 0.9


class Agent:
    def __init__(self, agentId, environment_shape, actions_number):
        self.id = agentId
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
            keras.layers.Dense(20, activation='relu', input_shape=environment_shape),
            keras.layers.Dense(10, activation='relu'),
            keras.layers.Dropout(rate=0.3),
            keras.layers.Dense(actions_number, activation="linear")
        ])
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=learning_rate))
        return model

    def act(self, environment_state):       
        self.previous_state = environment_state
        if self.gamma < np.random.rand():
            self.previous_action = tf.math.argmax(self.action_model(environment_state)[0])
        else:
            self.previous_action = np.random.randint(0, self.actions_number)
        return self.previous_action

    def train(self):
        if self.iteration_counter > BATCH_SIZE and self.iteration_counter % UPDATE_ACTION_MODEL == 0:
            if BATCH_SIZE >= self.replay_buffer.real_length:
                current_states, future_states, actions, rewards, done_flags = \
                    self.replay_buffer.get_experience(BATCH_SIZE)
                current_q_values = self.action_model.predict(current_states)
                for i in range(actions.size):
                    if done_flags[i]:
                        new_q_value = rewards[i]
                    else:
                        future_q_values = self.train_model.predict(future_states)
                        max_q_value = np.max(future_q_values)
                        new_q_value = rewards[i] + DISCOUNT_FACTOR * max_q_value
                    current_q_values[i][actions[i]] = new_q_value
                self.action_model.fit(current_states, current_q_values,
                                      batch_size=BATCH_SIZE, verbose=0, shuffle=False)

            if self.iteration_counter % UPDATE_TRAIN_MODEL == 0:
                self.train_model.set_weights(self.action_model.get_weights())
        self.iteration_counter += 1
        if self.gamma > GAMMA_MINIMUM:
            self.gamma -= GAMMA_DECAY

    def remember(self, next_state, reward, done_flag):
        self.replay_buffer.remember(self.previous_state, next_state, self.previous_action, reward, done_flag)


if __name__ == "__main__":
    pass
