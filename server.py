import grpc
from grpcService import data_pb2, data_pb2_grpc
import logging
import numpy as np
from random import randint

from concurrent import futures

from agent.Agent import Agent

num_agents = 0

class Servicer(data_pb2_grpc.LearnBoarServicer):

    def CreateAgent(self, request, context):
        global num_agents
        num_agents += 1
        env_shape = [request.env_shape]
        num_actions = request.num_actions
        self.env_shape = env_shape
        self.agent = Agent(num_agents, env_shape, num_actions)
        return data_pb2.AgentId(id=1)


    def SendData(self, request, context):
        env_state = np.empty(self.env_shape)
        env_state[0] = request.food_x
        env_state[1] = request.food_z
        env_state[2] = request.hp
        env_state[3] = request.satiety

        env_state = np.array([env_state])

        if self.agent.previous_action != None:
            self.agent.remember(env_state, request.reward, not request.isAlive)
            self.agent.train()

        return data_pb2.Action(action=int(self.agent.act(env_state)))

    
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    data_pb2_grpc.add_LearnBoarServicer_to_server(
        Servicer(),
        server
    )
    server.add_insecure_port('127.0.0.1:5000')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
