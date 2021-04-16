"""Define distributed kvstore"""

import os
import numpy as np

from . import rpc
from .graph_partition_book import NodePartitionPolicy, EdgePartitionPolicy
from .standalone_kvstore import KVClient as SA_KVClient

from .. import backend as F
from .. import utils
from .._ffi.ndarray import empty_shared_mem

############################ Register KVStore Requsts and Responses ###############################

KVSTORE_PULL = 901231

class PullResponse(rpc.Response):
    """Send the sliced data tensor back to the client.

    Parameters
    ----------
    server_id : int
        ID of current server
    data_tensor : tensor
        sliced data tensor
    """
    def __init__(self, server_id, data_tensor):
        self.server_id = server_id
        self.data_tensor = data_tensor

    def __getstate__(self):
        return self.server_id, self.data_tensor

    def __setstate__(self, state):
        self.server_id, self.data_tensor = state

class PullRequest(rpc.Request):
    """Send ID tensor to server and get target data tensor as response.

    Parameters
    ----------
    name : str
        data name
    id_tensor : tensor
        a vector storing the data ID
    """
    def __init__(self, name, id_tensor):
        self.name = name
        self.id_tensor = id_tensor

    def __getstate__(self):
        return self.name, self.id_tensor

    def __setstate__(self, state):
        self.name, self.id_tensor = state

    def process_request(self, server_state):
        kv_store = server_state.kv_store
        if self.name not in kv_store.part_policy:
            raise RuntimeError("KVServer cannot find partition policy with name: %s" % self.name)
        if self.name not in kv_store.data_store:
            raise RuntimeError("KVServer Cannot find data tensor with name: %s" % self.name)
        local_id = kv_store.part_policy[self.name].to_local(self.id_tensor)
        data = kv_store.pull_handlers[self.name](kv_store.data_store, self.name, local_id)
        res = PullResponse(kv_store.server_id, data)
        return res

KVSTORE_PUSH = 901232

class PushRequest(rpc.Request):
    """Send ID tensor and data tensor to server and update kvstore's data.

    This request has no response.

    Parameters
    ----------
    name : str
        data name
    id_tensor : tensor
        a vector storing the data ID
    data_tensor : tensor
        a tensor with the same row size of data ID
    """
    def __init__(self, name, id_tensor, data_tensor):
        self.name = name
        self.id_tensor = id_tensor
        self.data_tensor = data_tensor

    def __getstate__(self):
        return self.name, self.id_tensor, self.data_tensor

    def __setstate__(self, state):
        self.name, self.id_tensor, self.data_tensor = state

    def process_request(self, server_state):
        kv_store = server_state.kv_store
        if self.name not in kv_store.part_policy:
            raise RuntimeError("KVServer cannot find partition policy with name: %s" % self.name)
        if self.name not in kv_store.data_store:
            raise RuntimeError("KVServer Cannot find data tensor with name: %s" % self.name)
        local_id = kv_store.part_policy[self.name].to_local(self.id_tensor)
        kv_store.push_handlers[self.name](kv_store.data_store, self.name,
                                          local_id, self.data_tensor)

INIT_DATA = 901233
INIT_MSG = 'Init'

class InitDataResponse(rpc.Response):
    """Send a confirmation response (just a short string message) of
    InitDataRequest to client.

    Parameters
    ----------
    msg : string
        string message
    """
    def __init__(self, msg):
        self.msg = msg

    def __getstate__(self):
        return self.msg

    def __setstate__(self, state):
        self.msg = state

class InitDataRequest(rpc.Request):
    """Send meta data to server and init data tensor
    on server using UDF init function.

    Parameters
    ----------
    name : str
        data name
    shape : tuple
        data shape
    dtype : str
        data type string, e.g., 'int64', 'float32', etc.
    policy_str : str
        partition-policy string, e.g., 'edge' or 'node'.
    init_func : function
        UDF init function.
    """
    def __init__(self, name, shape, dtype, policy_str, init_func):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.policy_str = policy_str
        self.init_func = init_func

    def __getstate__(self):
        return self.name, self.shape, self.dtype, self.policy_str, self.init_func

    def __setstate__(self, state):
        self.name, self.shape, self.dtype, self.policy_str, self.init_func = state

    def process_request(self, server_state):
        kv_store = server_state.kv_store
        dtype = F.data_type_dict[self.dtype]

        # We should see requests from multiple clients. We need to ignore the duplicated
        # reqeusts.
        if self.name in kv_store.data_store:
            assert tuple(F.shape(kv_store.data_store[self.name])) == tuple(self.shape)
            assert F.reverse_data_type_dict[F.dtype(kv_store.data_store[self.name])] == self.dtype
            assert kv_store.part_policy[self.name].policy_str == self.policy_str
        else:
            if not kv_store.is_backup_server():
                data_tensor = self.init_func(self.shape, dtype)
                kv_store.init_data(name=self.name,
                                   policy_str=self.policy_str,
                                   data_tensor=data_tensor)
            else:
                kv_store.init_data(name=self.name, policy_str=self.policy_str)
        res = InitDataResponse(INIT_MSG)
        return res

BARRIER = 901234
BARRIER_MSG = 'Barrier'

class BarrierResponse(rpc.Response):
    """Send an confimation signal (just a short string message) of
    BarrierRequest to client.

    Parameters
    ----------
    msg : string
        string msg
    """
    def __init__(self, msg):
        self.msg = msg

    def __getstate__(self):
        return self.msg

    def __setstate__(self, state):
        self.msg = state

class BarrierRequest(rpc.Request):
    """Send a barrier signal (just a short string message) to server.

    Parameters
    ----------
    role : string
        client role
    """
    def __init__(self, role):
        self.role = role

    def __getstate__(self):
        return self.role

    def __setstate__(self, state):
        self.role = state

    def process_request(self, server_state):
        kv_store = server_state.kv_store
        role = server_state.roles
        count = kv_store.barrier_count[self.role]
        kv_store.barrier_count[self.role] = count + 1
        if kv_store.barrier_count[self.role] == len(role[self.role]):
            kv_store.barrier_count[self.role] = 0
            res_list = []
            for client_id, _ in role[self.role]:
                res_list.append((client_id, BarrierResponse(BARRIER_MSG)))
            return res_list
        return None

REGISTER_PULL = 901235
REGISTER_PULL_MSG = 'Register_Pull'

class RegisterPullHandlerResponse(rpc.Response):
    """Send a confirmation signal (just a short string message) of
    RegisterPullHandler to client.

    Parameters
    ----------
    msg : string
        string message
    """
    def __init__(self, msg):
        self.msg = msg

    def __getstate__(self):
        return self.msg

    def __setstate__(self, state):
        self.msg = state

class RegisterPullHandlerRequest(rpc.Request):
    """Send an UDF and register Pull handler on server.

    Parameters
    ----------
    pull_func : func
        UDF pull handler
    """
    def __init__(self, name, pull_func):
        self.name = name
        self.pull_func = pull_func

    def __getstate__(self):
        return self.name, self.pull_func

    def __setstate__(self, state):
        self.name, self.pull_func = state

    def process_request(self, server_state):
        kv_store = server_state.kv_store
        kv_store.pull_handlers[self.name] = self.pull_func
        res = RegisterPullHandlerResponse(REGISTER_PULL_MSG)
        return res

REGISTER_PUSH = 901236
REGISTER_PUSH_MSG = 'Register_Push'

class RegisterPushHandlerResponse(rpc.Response):
    """Send a confirmation signal (just a short string message) of
    RegisterPushHandler to client.

    Parameters
    ----------
    msg : string
        string message
    """
    def __init__(self, msg):
        self.msg = msg

    def __getstate__(self):
        return self.msg

    def __setstate__(self, state):
        self.msg = state

class RegisterPushHandlerRequest(rpc.Request):
    """Send an UDF to register Push handler on server.

    Parameters
    ----------
    push_func : func
        UDF push handler
    """
    def __init__(self, name, push_func):
        self.name = name
        self.push_func = push_func

    def __getstate__(self):
        return self.name, self.push_func

    def __setstate__(self, state):
        self.name, self.push_func = state

    def process_request(self, server_state):
        kv_store = server_state.kv_store
        kv_store.push_handlers[self.name] = self.push_func
        res = RegisterPushHandlerResponse(REGISTER_PUSH_MSG)
        return res

GET_SHARED = 901237
GET_SHARED_MSG = 'Get_Shared'

class GetSharedDataResponse(rpc.Response):
    """Send meta data of shared-memory tensor to client.

    Parameters
    ----------
    meta : dict
        a dict of meta, e.g.,

        {'data_0' : (shape, dtype, policy_str),
         'data_1' : (shape, dtype, policy_str)}
    """
    def __init__(self, meta):
        self.meta = meta

    def __getstate__(self):
        return self.meta

    def __setstate__(self, state):
        self.meta = state

class GetSharedDataRequest(rpc.Request):
    """Send a signal (just a short string message) to get the
    meta data of shared-tensor from server.

    Parameters
    ----------
    msg : string
        string message
    """
    def __init__(self, msg):
        self.msg = msg

    def __getstate__(self):
        return self.msg

    def __setstate__(self, state):
        self.msg = state

    def process_request(self, server_state):
        assert self.msg == GET_SHARED_MSG
        meta = {}
        kv_store = server_state.kv_store
        for name, data in kv_store.data_store.items():
            meta[name] = (F.shape(data),
                          F.reverse_data_type_dict[F.dtype(data)],
                          kv_store.part_policy[name].policy_str)
        res = GetSharedDataResponse(meta)
        return res

GET_PART_SHAPE = 901238

class GetPartShapeResponse(rpc.Response):
    """Send the partitioned data shape back to client.

    Parameters
    ----------
    shape : tuple
        shape of tensor
    """
    def __init__(self, shape):
        self.shape = shape

    def __getstate__(self):
        return self.shape

    def __setstate__(self, state):
        # When the shape has only one dimension, state is an integer.
        if isinstance(state, int):
            self.shape = (state,)
        else:
            self.shape = state

class GetPartShapeRequest(rpc.Request):
    """Send data name to get the partitioned data shape from server.

    Parameters
    ----------
    name : str
        data name
    """
    def __init__(self, name):
        self.name = name

    def __getstate__(self):
        return self.name

    def __setstate__(self, state):
        self.name = state

    def process_request(self, server_state):
        kv_store = server_state.kv_store
        if self.name not in kv_store.data_store:
            raise RuntimeError("KVServer Cannot find data tensor with name: %s" % self.name)
        data_shape = F.shape(kv_store.data_store[self.name])
        res = GetPartShapeResponse(data_shape)
        return res

SEND_META_TO_BACKUP = 901239
SEND_META_TO_BACKUP_MSG = "Send_Meta_TO_Backup"

class SendMetaToBackupResponse(rpc.Response):
    """Send a confirmation signal (just a short string message)
    of SendMetaToBackupRequest to client.
    """
    def __init__(self, msg):
        self.msg = msg

    def __getstate__(self):
        return self.msg

    def __setstate__(self, state):
        self.msg = state

class SendMetaToBackupRequest(rpc.Request):
    """Send meta data to backup server and backup server
    will use this meta data to read shared-memory tensor.

    Parameters
    ----------
    name : str
        data name
    dtype : str
        data type string
    shape : tuple of int
        data shape
    policy_str : str
        partition-policy string, e.g., 'edge' or 'node'.
    pull_handler : callable
        The callback function when data is pulled from kvstore.
    push_handler : callable
        The callback function when data is pushed to kvstore.
    """
    def __init__(self, name, dtype, shape, policy_str, pull_handler, push_handler):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.policy_str = policy_str
        self.pull_handler = pull_handler
        self.push_handler = push_handler

    def __getstate__(self):
        return self.name, self.dtype, self.shape, self.policy_str, self.pull_handler, \
                self.push_handler

    def __setstate__(self, state):
        self.name, self.dtype, self.shape, self.policy_str, self.pull_handler, \
                self.push_handler = state

    def process_request(self, server_state):
        kv_store = server_state.kv_store
        assert kv_store.is_backup_server()
        if self.name not in kv_store.data_store:
            shared_data = empty_shared_mem(self.name+'-kvdata-', False, self.shape, self.dtype)
            dlpack = shared_data.to_dlpack()
            kv_store.data_store[self.name] = F.zerocopy_from_dlpack(dlpack)
            kv_store.part_policy[self.name] = kv_store.find_policy(self.policy_str)
            kv_store.pull_handlers[self.name] = self.pull_handler
            kv_store.push_handlers[self.name] = self.push_handler
        else:
            assert tuple(F.shape(kv_store.data_store[self.name])) == tuple(self.shape)
            assert F.reverse_data_type_dict[F.dtype(kv_store.data_store[self.name])] == self.dtype
            assert kv_store.part_policy[self.name].policy_str == self.policy_str
            assert kv_store.pull_handlers[self.name] == self.pull_handler
            assert kv_store.push_handlers[self.name] == self.push_handler
        res = SendMetaToBackupResponse(SEND_META_TO_BACKUP_MSG)
        return res

DELETE_DATA = 901240
DELETE_MSG = "Delete_Data"

class DeleteDataResponse(rpc.Response):
    """Send a confirmation signal (just a short string message)
    of DeleteDataRequest to client.
    """
    def __init__(self, msg):
        self.msg = msg

    def __getstate__(self):
        return self.msg

    def __setstate__(self, state):
        self.msg = state

class DeleteDataRequest(rpc.Request):
    """Send message to server to delete data tensor

    Parameters
    ----------
    name : str
        data name
    """
    def __init__(self, name):
        self.name = name

    def __getstate__(self):
        return self.name

    def __setstate__(self, state):
        self.name = state

    def process_request(self, server_state):
        kv_store = server_state.kv_store
        if self.name in kv_store.data_store:
            del kv_store.data_store[self.name]
            del kv_store.part_policy[self.name]
            del kv_store.push_handlers[self.name]
            del kv_store.pull_handlers[self.name]
        res = DeleteDataResponse(DELETE_MSG)
        return res

############################ KVServer ###############################

def default_push_handler(target, name, id_tensor, data_tensor):
    """Default handler for PUSH message.

    On default, _push_handler perform scatter_row() operation for the tensor.

    Parameters
    ----------
    target : tensor
        target tensor
    name : str
        data name
    id_tensor : tensor
        a vector storing the ID list.
    data_tensor : tensor
        a tensor with the same row size of id
    """
    # TODO(chao): support Tensorflow backend
    target[name][id_tensor] = data_tensor

def default_pull_handler(target, name, id_tensor):
    """Default handler for PULL operation.

    On default, _pull_handler perform gather_row() operation for the tensor.

    Parameters
    ----------
    target : tensor
        target tensor
    name : str
        data name
    id_tensor : tensor
        a vector storing the ID list.

    Return
    ------
    tensor
        a tensor with the same row size of ID.
    """
    # TODO(chao): support Tensorflow backend
    return target[name][id_tensor]

class KVServer(object):
    """KVServer is a lightweight key-value store service for DGL distributed training.

    In practice, developers can use KVServer to hold large-scale graph features or
    graph embeddings across machines in a distributed setting. KVServer depends on DGL rpc
    infrastructure thats support backup servers, which means we can lunach many KVServers
    on the same machine for load-balancing.

    DO NOT use KVServer in mult-threads because this behavior is not defined. For now, KVServer
    can only support CPU-to-CPU communication. We may support GPU-communication in the future.

    Parameters
    ----------
    server_id : int
        ID of current server (starts from 0).
    ip_config : str
        Path of IP configuration file.
    num_servers : int
        Server count on each machine.
    num_clients : int
        Total number of KVClients that will be connected to the KVServer.
    """
    def __init__(self, server_id, ip_config, num_servers, num_clients):
        assert server_id >= 0, 'server_id (%d) cannot be a negative number.' % server_id
        assert num_servers > 0, 'num_servers (%d) must be a positive number.' % num_servers
        assert os.path.exists(ip_config), 'Cannot open file: %s' % ip_config
        assert num_clients >= 0, 'num_clients (%d) cannot be a negative number.' % num_clients
        # Register services on server
        rpc.register_service(KVSTORE_PULL,
                             PullRequest,
                             PullResponse)
        rpc.register_service(KVSTORE_PUSH,
                             PushRequest,
                             None)
        rpc.register_service(INIT_DATA,
                             InitDataRequest,
                             InitDataResponse)
        rpc.register_service(BARRIER,
                             BarrierRequest,
                             BarrierResponse)
        rpc.register_service(REGISTER_PUSH,
                             RegisterPushHandlerRequest,
                             RegisterPushHandlerResponse)
        rpc.register_service(REGISTER_PULL,
                             RegisterPullHandlerRequest,
                             RegisterPullHandlerResponse)
        rpc.register_service(GET_SHARED,
                             GetSharedDataRequest,
                             GetSharedDataResponse)
        rpc.register_service(GET_PART_SHAPE,
                             GetPartShapeRequest,
                             GetPartShapeResponse)
        rpc.register_service(SEND_META_TO_BACKUP,
                             SendMetaToBackupRequest,
                             SendMetaToBackupResponse)
        rpc.register_service(DELETE_DATA,
                             DeleteDataRequest,
                             DeleteDataResponse)
        # Store the tensor data with specified data name
        self._data_store = {}
        # Store the partition information with specified data name
        self._policy_set = set()
        self._part_policy = {}
        # Basic information
        self._server_id = server_id
        self._server_namebook = rpc.read_ip_config(ip_config, num_servers)
        assert server_id in self._server_namebook, \
                'Trying to start server {}, but there are {} servers in the config file'.format(
                    server_id, len(self._server_namebook))
        self._machine_id = self._server_namebook[server_id][0]
        self._group_count = self._server_namebook[server_id][3]
        # We assume partition_id is equal to machine_id
        self._part_id = self._machine_id
        self._num_clients = num_clients
        self._barrier_count = {}
        # push and pull handler
        self._push_handlers = {}
        self._pull_handlers = {}

    @property
    def server_id(self):
        """Get server ID"""
        return self._server_id

    @property
    def barrier_count(self):
        """Get barrier count"""
        return self._barrier_count

    @barrier_count.setter
    def barrier_count(self, count):
        """Set barrier count"""
        self._barrier_count = count

    @property
    def num_clients(self):
        """Get number of clients"""
        return self._num_clients

    @property
    def data_store(self):
        """Get data store"""
        return self._data_store

    @property
    def part_policy(self):
        """Get part policy"""
        return self._part_policy

    @property
    def part_id(self):
        """Get part ID"""
        return self._part_id

    @property
    def push_handlers(self):
        """Get push handler"""
        return self._push_handlers

    @property
    def pull_handlers(self):
        """Get pull handler"""
        return self._pull_handlers

    def is_backup_server(self):
        """Return True if current server is a backup server.
        """
        if self._server_id % self._group_count == 0:
            return False
        return True

    def add_part_policy(self, policy):
        """Add partition policy to kvserver.

        Parameters
        ----------
        policy : PartitionPolicy
            Store the partition information
        """
        self._policy_set.add(policy)

    def init_data(self, name, policy_str, data_tensor=None):
        """Init data tensor on kvserver.

        Parameters
        ----------
        name : str
            data name
        policy_str : str
            partition-policy string, e.g., 'edge' or 'node'.
        data_tensor : tensor
            If the data_tensor is None, KVServer will
            read shared-memory when client invoking get_shared_data().
        """
        assert len(name) > 0, 'name cannot be empty.'
        if name in self._data_store:
            raise RuntimeError("Data %s has already exists!" % name)
        self._part_policy[name] = self.find_policy(policy_str)
        if data_tensor is not None: # Create shared-tensor
            data_type = F.reverse_data_type_dict[F.dtype(data_tensor)]
            shared_data = empty_shared_mem(name+'-kvdata-', True, data_tensor.shape, data_type)
            dlpack = shared_data.to_dlpack()
            self._data_store[name] = F.zerocopy_from_dlpack(dlpack)
            rpc.copy_data_to_shared_memory(self._data_store[name], data_tensor)
            assert self._part_policy[name].get_part_size() == data_tensor.shape[0], \
                    'kvserver expect partition {} for {} has {} rows, but gets {} rows'.format(
                        self._part_policy[name].part_id,
                        policy_str,
                        self._part_policy[name].get_part_size(),
                        data_tensor.shape[0])
        self._pull_handlers[name] = default_pull_handler
        self._push_handlers[name] = default_push_handler

    def find_policy(self, policy_str):
        """Find a partition policy from existing policy set

        Parameters
        ----------
        policy_str : str
            partition-policy string, e.g., 'edge' or 'node'.
        """
        for policy in self._policy_set:
            if policy_str == policy.policy_str:
                return policy
        raise RuntimeError("Cannot find policy_str: %s from kvserver." % policy_str)

############################ KVClient ###############################

class KVClient(object):
    """KVClient is used to push/pull data to/from KVServer. If the
    target kvclient and kvserver are in the same machine, they can
    communicate with each other using local shared-memory
    automatically, instead of going through the tcp/ip RPC.

    DO NOT use KVClient in multi-threads because this behavior is
    not defined. For now, KVClient can only support CPU-to-CPU communication.
    We may support GPU-communication in the future.

    Parameters
    ----------
    ip_config : str
        Path of IP configuration file.
    num_servers : int
        Server count on each machine.
    role : str
        We can set different role for kvstore.
    """
    def __init__(self, ip_config, num_servers, role='default'):
        assert rpc.get_rank() != -1, \
                'Please invoke rpc.connect_to_server() before creating KVClient.'
        assert os.path.exists(ip_config), 'Cannot open file: %s' % ip_config
        assert num_servers > 0, 'num_servers (%d) must be a positive number.' % num_servers
        # Register services on client
        rpc.register_service(KVSTORE_PULL,
                             PullRequest,
                             PullResponse)
        rpc.register_service(KVSTORE_PUSH,
                             PushRequest,
                             None)
        rpc.register_service(INIT_DATA,
                             InitDataRequest,
                             InitDataResponse)
        rpc.register_service(BARRIER,
                             BarrierRequest,
                             BarrierResponse)
        rpc.register_service(REGISTER_PUSH,
                             RegisterPushHandlerRequest,
                             RegisterPushHandlerResponse)
        rpc.register_service(REGISTER_PULL,
                             RegisterPullHandlerRequest,
                             RegisterPullHandlerResponse)
        rpc.register_service(GET_SHARED,
                             GetSharedDataRequest,
                             GetSharedDataResponse)
        rpc.register_service(GET_PART_SHAPE,
                             GetPartShapeRequest,
                             GetPartShapeResponse)
        rpc.register_service(SEND_META_TO_BACKUP,
                             SendMetaToBackupRequest,
                             SendMetaToBackupResponse)
        rpc.register_service(DELETE_DATA,
                             DeleteDataRequest,
                             DeleteDataResponse)
        # Store the tensor data with specified data name
        self._data_store = {}
        # Store the partition information with specified data name
        self._part_policy = {}
        # This stores all unique partition policies in the kvstore. The key is the policy name.
        self._all_possible_part_policy = {}
        # Store the full data shape across kvserver
        self._full_data_shape = {}
        # Store all the data name
        self._data_name_list = set()
        # Basic information
        self._server_namebook = rpc.read_ip_config(ip_config, num_servers)
        self._server_count = len(self._server_namebook)
        self._group_count = self._server_namebook[0][3]
        self._machine_count = int(self._server_count / self._group_count)
        self._client_id = rpc.get_rank()
        self._machine_id = rpc.get_machine_id()
        self._part_id = self._machine_id
        self._main_server_id = self._machine_id * self._group_count
        # push and pull handler
        self._pull_handlers = {}
        self._push_handlers = {}
        # register role on server-0
        self._role = role

    @property
    def all_possible_part_policy(self):
        """Get all possible partition policies"""
        return self._all_possible_part_policy

    @property
    def client_id(self):
        """Get client ID"""
        return self._client_id

    @property
    def role(self):
        """Get client role"""
        return self._role

    @property
    def machine_id(self):
        """Get machine ID"""
        return self._machine_id

    @property
    def num_servers(self):
        """Get the number of servers"""
        return self._server_count

    def barrier(self):
        """Barrier for all client nodes.

        This API will be blocked untill all the clients invoke this API.
        """
        request = BarrierRequest(self._role)
        rpc.send_request(0, request)
        response = rpc.recv_response()
        assert response.msg == BARRIER_MSG

    def register_push_handler(self, name, func):
        """Register UDF push function.

        This UDF is triggered for every push. The signature of the UDF is

        ```
        def push_handler(data_store, name, local_offset, data)
        ```

        ``data_store`` is a dict that contains all tensors in the kvstore. ``name`` is the name
        of the tensor where new data is pushed to. ``local_offset`` is the offset where new
        data should be written in the tensor in the local partition. ``data`` is the new data
        to be written.

        Parameters
        ----------
        name : str
            The name of the tensor
        func : callable
            The function to be called.
        """
        self.barrier()
        request = RegisterPushHandlerRequest(name, func)
        # send request to all the server nodes
        for server_id in range(self._server_count):
            rpc.send_request(server_id, request)
        # recv response from all the server nodes
        for _ in range(self._server_count):
            response = rpc.recv_response()
            assert response.msg == REGISTER_PUSH_MSG
        self._push_handlers[name] = func
        self.barrier()

    def register_pull_handler(self, name, func):
        """Register UDF pull function.

        This UDF is triggered for every pull. The signature of the UDF is

        ```
        def pull_handler(data_store, name, local_offset)
        ```

        ``data_store`` is a dict that contains all tensors in the kvstore. ``name`` is the name
        of the tensor where new data is pushed to. ``local_offset`` is the offset where new
        data should be written in the tensor in the local partition.

        Parameters
        ----------
        name : str
            The name of the tensor
        func : callable
            The function to be called.
        """
        self.barrier()
        request = RegisterPullHandlerRequest(name, func)
        # send request to all the server nodes
        for server_id in range(self._server_count):
            rpc.send_request(server_id, request)
        # recv response from all the server nodes
        for _ in range(self._server_count):
            response = rpc.recv_response()
            assert response.msg == REGISTER_PULL_MSG
        self._pull_handlers[name] = func
        self.barrier()

    def init_data(self, name, shape, dtype, part_policy, init_func):
        """Send message to kvserver to initialize new data tensor and mapping this
        data from server side to client side.

        Parameters
        ----------
        name : str
            data name
        shape : list or tuple of int
            data shape
        dtype : dtype
            data type
        part_policy : PartitionPolicy
            partition policy.
        init_func : func
            UDF init function
        """
        assert len(name) > 0, 'name cannot be empty.'
        assert len(shape) > 0, 'shape cannot be empty'
        assert name not in self._data_name_list, 'data name: %s already exists.' % name
        self.barrier()
        shape = list(shape)

        # Send request to the servers to initialize data.
        # The servers may handle the duplicated initializations.
        part_shape = shape.copy()
        part_shape[0] = part_policy.get_part_size()
        request = InitDataRequest(name,
                                  tuple(part_shape),
                                  F.reverse_data_type_dict[dtype],
                                  part_policy.policy_str,
                                  init_func)
        # The request is sent to the servers in one group, which are on the same machine.
        for n in range(self._group_count):
            server_id = part_policy.part_id * self._group_count + n
            rpc.send_request(server_id, request)
        for _ in range(self._group_count):
            response = rpc.recv_response()
            assert response.msg == INIT_MSG

        self.barrier()
        # Create local shared-data
        local_shape = shape.copy()
        local_shape[0] = part_policy.get_part_size()
        if name in self._part_policy:
            raise RuntimeError("Policy %s has already exists!" % name)
        if name in self._data_store:
            raise RuntimeError("Data %s has already exists!" % name)
        if name in self._full_data_shape:
            raise RuntimeError("Data shape %s has already exists!" % name)
        self._part_policy[name] = part_policy
        self._all_possible_part_policy[part_policy.policy_str] = part_policy
        shared_data = empty_shared_mem(name+'-kvdata-', False, \
            local_shape, F.reverse_data_type_dict[dtype])
        dlpack = shared_data.to_dlpack()
        self._data_store[name] = F.zerocopy_from_dlpack(dlpack)
        self._data_name_list.add(name)
        self._full_data_shape[name] = tuple(shape)
        self._pull_handlers[name] = default_pull_handler
        self._push_handlers[name] = default_push_handler

        # Now we need to tell the backup server the new tensor.
        request = SendMetaToBackupRequest(name, F.reverse_data_type_dict[dtype],
                                          part_shape, part_policy.policy_str,
                                          self._pull_handlers[name],
                                          self._push_handlers[name])
        # send request to all the backup server nodes
        for i in range(self._group_count-1):
            server_id = self._machine_id * self._group_count + i + 1
            rpc.send_request(server_id, request)
        # recv response from all the backup server nodes
        for _ in range(self._group_count-1):
            response = rpc.recv_response()
            assert response.msg == SEND_META_TO_BACKUP_MSG
        self.barrier()

    def delete_data(self, name):
        """Send message to kvserver to delete tensor and clear the meta data

        Parameters
        ----------
        name : str
            data name
        """
        assert len(name) > 0, 'name cannot be empty.'
        assert name in self._data_name_list, 'data name: %s not exists.' % name
        self.barrier()
        part_policy = self._part_policy[name]

        # send request to every server nodes
        request = DeleteDataRequest(name)
        for n in range(self._group_count):
            server_id = part_policy.part_id * self._group_count + n
            rpc.send_request(server_id, request)
        for _ in range(self._group_count):
            response = rpc.recv_response()
            assert response.msg == DELETE_MSG

        self.barrier()
        self._data_name_list.remove(name)
        # TODO(chao) : remove the delete log print
        del self._data_store[name]
        del self._full_data_shape[name]
        del self._part_policy[name]
        del self._pull_handlers[name]
        del self._push_handlers[name]
        self.barrier()

    def map_shared_data(self, partition_book):
        """Mapping shared-memory tensor from server to client.

        Parameters
        ----------
        partition_book : GraphPartitionBook
            Store the partition information
        """
        # Get all partition policies
        for ntype in partition_book.ntypes:
            policy = NodePartitionPolicy(partition_book, ntype)
            self._all_possible_part_policy[policy.policy_str] = policy
        for etype in partition_book.etypes:
            policy = EdgePartitionPolicy(partition_book, etype)
            self._all_possible_part_policy[policy.policy_str] = policy

        # Get shared data from server side
        self.barrier()
        request = GetSharedDataRequest(GET_SHARED_MSG)
        rpc.send_request(self._main_server_id, request)
        response = rpc.recv_response()
        for name, meta in response.meta.items():
            if name not in self._data_name_list:
                shape, dtype, policy_str = meta
                assert policy_str in self._all_possible_part_policy
                shared_data = empty_shared_mem(name+'-kvdata-', False, shape, dtype)
                dlpack = shared_data.to_dlpack()
                self._data_store[name] = F.zerocopy_from_dlpack(dlpack)
                self._part_policy[name] = self._all_possible_part_policy[policy_str]
                self._pull_handlers[name] = default_pull_handler
                self._push_handlers[name] = default_push_handler
        # Get full data shape across servers
        for name, meta in response.meta.items():
            if name not in self._data_name_list:
                shape, _, _ = meta
                data_shape = list(shape)
                data_shape[0] = 0
                request = GetPartShapeRequest(name)
                # send request to all main server nodes
                for machine_id in range(self._machine_count):
                    server_id = machine_id * self._group_count
                    rpc.send_request(server_id, request)
                # recv response from all the main server nodes
                for _ in range(self._machine_count):
                    res = rpc.recv_response()
                    data_shape[0] += res.shape[0]
                self._full_data_shape[name] = tuple(data_shape)
        # Send meta data to backup servers
        for name, meta in response.meta.items():
            shape, dtype, policy_str = meta
            request = SendMetaToBackupRequest(name, dtype, shape, policy_str,
                                              self._pull_handlers[name],
                                              self._push_handlers[name])
            # send request to all the backup server nodes
            for i in range(self._group_count-1):
                server_id = self._machine_id * self._group_count + i + 1
                rpc.send_request(server_id, request)
            # recv response from all the backup server nodes
            for _ in range(self._group_count-1):
                response = rpc.recv_response()
                assert response.msg == SEND_META_TO_BACKUP_MSG
            self._data_name_list.add(name)
        self.barrier()

    def data_name_list(self):
        """Get all the data name"""
        return list(self._data_name_list)

    def get_data_meta(self, name):
        """Get meta data (data_type, data_shape, partition_policy)
        """
        assert len(name) > 0, 'name cannot be empty.'
        data_type = F.dtype(self._data_store[name])
        data_shape = self._full_data_shape[name]
        part_policy = self._part_policy[name]
        return (data_type, data_shape, part_policy)

    def push(self, name, id_tensor, data_tensor):
        """Push data to KVServer.

        Note that, the push() is an non-blocking operation that will return immediately.

        Parameters
        ----------
        name : str
            data name
        id_tensor : tensor
            a vector storing the global data ID
        data_tensor : tensor
            a tensor with the same row size of data ID
        """
        assert len(name) > 0, 'name cannot be empty.'
        id_tensor = utils.toindex(id_tensor)
        id_tensor = id_tensor.tousertensor()
        assert F.ndim(id_tensor) == 1, 'ID must be a vector.'
        assert F.shape(id_tensor)[0] == F.shape(data_tensor)[0], \
        'The data must has the same row size with ID.'
        # partition data
        machine_id = self._part_policy[name].to_partid(id_tensor)
        # sort index by machine id
        sorted_id = F.tensor(np.argsort(F.asnumpy(machine_id)))
        id_tensor = id_tensor[sorted_id]
        data_tensor = data_tensor[sorted_id]
        machine, count = np.unique(F.asnumpy(machine_id), return_counts=True)
        # push data to server by order
        start = 0
        local_id = None
        local_data = None
        for idx, machine_idx in enumerate(machine):
            end = start + count[idx]
            if start == end: # No data for target machine
                continue
            partial_id = id_tensor[start:end]
            partial_data = data_tensor[start:end]
            if machine_idx == self._machine_id: # local push
                # Note that DO NOT push local data right now because we can overlap
                # communication-local_push here
                local_id = self._part_policy[name].to_local(partial_id)
                local_data = partial_data
            else: # push data to remote server
                request = PushRequest(name, partial_id, partial_data)
                rpc.send_request_to_machine(machine_idx, request)
            start += count[idx]
        if local_id is not None: # local push
            self._push_handlers[name](self._data_store, name, local_id, local_data)

    def pull(self, name, id_tensor):
        """Pull message from KVServer.

        Parameters
        ----------
        name : str
            data name
        id_tensor : tensor
            a vector storing the ID list

        Returns
        -------
        tensor
            a data tensor with the same row size of id_tensor.
        """
        assert len(name) > 0, 'name cannot be empty.'
        id_tensor = utils.toindex(id_tensor)
        id_tensor = id_tensor.tousertensor()
        assert F.ndim(id_tensor) == 1, 'ID must be a vector.'
        if self._pull_handlers[name] is default_pull_handler: # Use fast-pull
            part_id = self._part_policy[name].to_partid(id_tensor)
            return rpc.fast_pull(name, id_tensor, part_id, KVSTORE_PULL,
                                 self._machine_count,
                                 self._group_count,
                                 self._machine_id,
                                 self._client_id,
                                 self._data_store[name],
                                 self._part_policy[name])
        else:
            # partition data
            machine_id = self._part_policy[name].to_partid(id_tensor)
            # sort index by machine id
            sorted_id = F.tensor(np.argsort(F.asnumpy(machine_id)))
            back_sorted_id = F.tensor(np.argsort(F.asnumpy(sorted_id)))
            id_tensor = id_tensor[sorted_id]
            machine, count = np.unique(F.asnumpy(machine_id), return_counts=True)
            # pull data from server by order
            start = 0
            pull_count = 0
            local_id = None
            for idx, machine_idx in enumerate(machine):
                end = start + count[idx]
                if start == end: # No data for target machine
                    continue
                partial_id = id_tensor[start:end]
                if machine_idx == self._machine_id: # local pull
                    # Note that DO NOT pull local data right now because we can overlap
                    # communication-local_pull here
                    local_id = self._part_policy[name].to_local(partial_id)
                else: # pull data from remote server
                    request = PullRequest(name, partial_id)
                    rpc.send_request_to_machine(machine_idx, request)
                    pull_count += 1
                start += count[idx]
            # recv response
            response_list = []
            if local_id is not None: # local pull
                local_data = self._pull_handlers[name](self._data_store, name, local_id)
                server_id = self._main_server_id
                local_response = PullResponse(server_id, local_data)
                response_list.append(local_response)
            # wait response from remote server nodes
            for _ in range(pull_count):
                remote_response = rpc.recv_response()
                response_list.append(remote_response)
            # sort response by server_id and concat tensor
            response_list.sort(key=self._take_id)
            data_tensor = F.cat(seq=[response.data_tensor for response in response_list], dim=0)
            return data_tensor[back_sorted_id] # return data with original index order

    def _take_id(self, elem):
        """Used by sort response list
        """
        return elem.server_id

KVCLIENT = None

def init_kvstore(ip_config, num_servers, role):
    """initialize KVStore"""
    global KVCLIENT
    if KVCLIENT is None:
        if os.environ.get('DGL_DIST_MODE', 'standalone') == 'standalone':
            KVCLIENT = SA_KVClient()
        else:
            KVCLIENT = KVClient(ip_config, num_servers, role)

def close_kvstore():
    """Close the current KVClient"""
    global KVCLIENT
    KVCLIENT = None

def get_kvstore():
    """get the KVClient"""
    return KVCLIENT
