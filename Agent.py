import tensorflow as tf
import numpy as np
from tensorflow import keras
from ReplayBuffer import ReplayBuffer


MEMORY_LENGTH = 10_000
BATCH_SIZE = 32
UPDATE_ACTION_MODEL = 4
UPDATE_TRAIN_MODEL = 1_000
class Agent:
    def __init__(self, environment_shape, actions_number):
        self.replay_buffer = ReplayBuffer(MEMORY_LENGTH, environment_shape)
        self.previous_state = None
        self.previous_action = None
        self.iteration_counter = 0

        # Being trained really often
        self.action_model = Agent.Create_model(environment_shape, actions_number)

        # Used to train other model. Being updated rarely
        self.train_model = Agent.Create_model(environment_shape, actions_number)
        self.train_model.set_weights(self.action_model.get_weights())

    @staticmethod
    def Create_model(environment_shape, actions_number, learning_rate=0.001):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(64, input_shape=environment_shape))
        model.add(keras.layers.Dense(64))
        model.add(keras.layers.Dense(actions_number, activation="linear"))
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=learning_rate))
        return model

    def Act(self, environment_state):
        self.previous_state = environment_state
        self.previous_action = tf.math.argmax(self.action_model(environment_state))
        return self.previous_action

    def Train(self):
        if self.iteration_counter % UPDATE_ACTION_MODEL == 0:
            if BATCH_SIZE >= self.replay_buffer.real_length:
                expirience = self.replay_buffer.GetExpirience(BATCH_SIZE)
                # continue training

            if self.iteration_counter % UPDATE_TRAIN_MODEL == 0:
                self.train_model.set_weights(self.action_model.get_weights())
        self.iteration_counter += 1

    def Remember(self, next_state, reward, done_flag):
        self.replay_buffer.Remember(self.previous_state, next_state, self.previous_action, reward, done_flag)
