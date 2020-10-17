import grpc
import data_pb2
import data_pb2_grpc
import logging
from random import randint

from concurrent import futures

class Servicer(data_pb2_grpc.LearnBoarServicer):

    def SendData(self, request, context):
        n = randint(0, 5)
        return data_pb2.Action(action=n)

    
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    data_pb2_grpc.add_LearnBoarServicer_to_server(
        Servicer(),
        server
    )
    server.add_insecure_port('127.0.0.1:50061')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
