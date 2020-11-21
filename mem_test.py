from agent import Agent
import numpy as np
agent = Agent.Agent(0, [1], 2)
done = False
for i in range(20):

    if agent.previous_action is not None:
        agent.remember(i, 10, i % 5 == 0)
        agent.train()
    agent.act(np.array([i]))
    done = i % 5 == 0
