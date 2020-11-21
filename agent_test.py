from agent import Agent
import gym
import numpy as np


env = gym.make('CartPole-v1')
agent = Agent.Agent(0, [4], 2)
for i_episode in range(2000):
    observation = env.reset()
    done = False
    for t in range(200):
        env.render()
        observation = np.array([observation])
        if agent.previous_action is not None:
            agent.remember(observation, reward, done)
            agent.train()
        action = int(agent.act(observation))
        observation, reward, done, info = env.step(action)
        if done:
            if i_episode % 50 == 0:
                print("{0} episode {1} timesteps, random is {2}".format(i_episode, t + 1, agent.gamma))
            break
env.close()
