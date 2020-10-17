import grpc
from grpcService import data_pb2, data_pb2_grpc


def run():
    with grpc.insecure_channel('127.0.0.1:50061') as channel:
        stub = data_pb2_grpc.LearnBoarStub(channel)
        response_agent = stub.CreateAgent(data_pb2.AgentData(
            env_shape = int(4),
            num_actions = int(4)
        ))
        for i in range(20):
            response_action = stub.SendData(data_pb2.EnvData(
                isAlive = True,
                food_x = 0.0,
                food_z = 0.0,
                hp = 100.0,
                satiety = 36.0,
                reward = int(1)
            ))
            print(i)


if __name__ == "__main__":
    run()