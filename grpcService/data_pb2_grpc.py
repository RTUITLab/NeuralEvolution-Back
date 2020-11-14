# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from grpcService import data_pb2 as data__pb2


class LearnBoarStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.SendData = channel.unary_unary(
                '/LearnBoar/SendData',
                request_serializer=data__pb2.EnvData.SerializeToString,
                response_deserializer=data__pb2.Action.FromString,
                )
        self.CreateAgent = channel.unary_unary(
                '/LearnBoar/CreateAgent',
                request_serializer=data__pb2.AgentData.SerializeToString,
                response_deserializer=data__pb2.AgentId.FromString,
                )


class LearnBoarServicer(object):
    """Missing associated documentation comment in .proto file."""

    def SendData(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CreateAgent(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_LearnBoarServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'SendData': grpc.unary_unary_rpc_method_handler(
                    servicer.SendData,
                    request_deserializer=data__pb2.EnvData.FromString,
                    response_serializer=data__pb2.Action.SerializeToString,
            ),
            'CreateAgent': grpc.unary_unary_rpc_method_handler(
                    servicer.CreateAgent,
                    request_deserializer=data__pb2.AgentData.FromString,
                    response_serializer=data__pb2.AgentId.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'LearnBoar', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class LearnBoar(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def SendData(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/LearnBoar/SendData',
            data__pb2.EnvData.SerializeToString,
            data__pb2.Action.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CreateAgent(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/LearnBoar/CreateAgent',
            data__pb2.AgentData.SerializeToString,
            data__pb2.AgentId.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
