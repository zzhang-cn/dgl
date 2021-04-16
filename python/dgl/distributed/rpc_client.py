"""Functions used by client."""

import os
import socket
import atexit

from . import rpc
from .constants import MAX_QUEUE_SIZE

if os.name != 'nt':
    import fcntl
    import struct

def local_ip4_addr_list():
    """Return a set of IPv4 address
    """
    assert os.name != 'nt', 'Do not support Windows rpc yet.'
    nic = set()
    for if_nidx in socket.if_nameindex():
        name = if_nidx[1]
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            ip_of_ni = fcntl.ioctl(sock.fileno(),
                                   0x8915,  # SIOCGIFADDR
                                   struct.pack('256s', name[:15].encode("UTF-8")))
        except OSError as e:
            if e.errno == 99: # EADDRNOTAVAIL
                print("Warning!",
                      "Interface: {}".format(name),
                      "IP address not available for interface.",
                      sep='\n')
                continue
            else:
                raise e

        ip_addr = socket.inet_ntoa(ip_of_ni[20:24])
        nic.add(ip_addr)
    return nic

def get_local_machine_id(server_namebook):
    """Given server_namebook, find local machine ID

    Parameters
    ----------
    server_namebook: dict
        IP address namebook of server nodes, where key is the server's ID
        (start from 0) and value is the server's machine_id, IP address,
        port, and group_count, e.g.,

          {0:'[0, '172.31.40.143', 30050, 2],
           1:'[0, '172.31.40.143', 30051, 2],
           2:'[1, '172.31.36.140', 30050, 2],
           3:'[1, '172.31.36.140', 30051, 2],
           4:'[2, '172.31.47.147', 30050, 2],
           5:'[2, '172.31.47.147', 30051, 2],
           6:'[3, '172.31.30.180', 30050, 2],
           7:'[3, '172.31.30.180', 30051, 2]}

    Returns
    -------
    int
        local machine ID
    """
    res = 0
    ip_list = local_ip4_addr_list()
    for _, data in server_namebook.items():
        machine_id = data[0]
        ip_addr = data[1]
        if ip_addr in ip_list:
            res = machine_id
            break
    return res

def get_local_usable_addr(probe_addr):
    """Get local usable IP and port

    Returns
    -------
    str
        IP address, e.g., '192.168.8.12:50051'
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # should get the address on the same subnet as probe_addr's
        sock.connect((probe_addr, 1))
        ip_addr = sock.getsockname()[0]
    except ValueError:
        ip_addr = '127.0.0.1'
    finally:
        sock.close()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    sock.close()

    return ip_addr + ':' + str(port)


def connect_to_server(ip_config, num_servers, max_queue_size=MAX_QUEUE_SIZE, net_type='socket'):
    """Connect this client to server.

    Parameters
    ----------
    ip_config : str
        Path of server IP configuration file.
    num_servers : int
        server count on each machine.
    max_queue_size : int
        Maximal size (bytes) of client queue buffer (~20 GB on default).
        Note that the 20 GB is just an upper-bound and DGL uses zero-copy and
        it will not allocate 20GB memory at once.
    net_type : str
        Networking type. Current options are: 'socket'.

    Raises
    ------
    ConnectionError : If anything wrong with the connection.
    """
    assert num_servers > 0, 'num_servers (%d) must be a positive number.' % num_servers
    assert max_queue_size > 0, 'queue_size (%d) cannot be a negative number.' % max_queue_size
    assert net_type in ('socket'), 'net_type (%s) can only be \'socket\'.' % net_type
    # Register some basic service
    rpc.register_service(rpc.CLIENT_REGISTER,
                         rpc.ClientRegisterRequest,
                         rpc.ClientRegisterResponse)
    rpc.register_service(rpc.SHUT_DOWN_SERVER,
                         rpc.ShutDownRequest,
                         None)
    rpc.register_service(rpc.GET_NUM_CLIENT,
                         rpc.GetNumberClientsRequest,
                         rpc.GetNumberClientsResponse)
    rpc.register_service(rpc.CLIENT_BARRIER,
                         rpc.ClientBarrierRequest,
                         rpc.ClientBarrierResponse)
    rpc.register_sig_handler()
    server_namebook = rpc.read_ip_config(ip_config, num_servers)
    num_servers = len(server_namebook)
    rpc.set_num_server(num_servers)
    # group_count means how many servers
    # (main_server + bakcup_server) in total inside a machine.
    group_count = []
    max_machine_id = 0
    for server_info in server_namebook.values():
        group_count.append(server_info[3])
        if server_info[0] > max_machine_id:
            max_machine_id = server_info[0]
    rpc.set_num_server_per_machine(group_count[0])
    num_machines = max_machine_id+1
    rpc.set_num_machines(num_machines)
    machine_id = get_local_machine_id(server_namebook)
    rpc.set_machine_id(machine_id)
    rpc.create_sender(max_queue_size, net_type)
    rpc.create_receiver(max_queue_size, net_type)
    # Get connected with all server nodes
    for server_id, addr in server_namebook.items():
        server_ip = addr[1]
        server_port = addr[2]
        rpc.add_receiver_addr(server_ip, server_port, server_id)
    rpc.sender_connect()
    # Get local usable IP address and port
    ip_addr = get_local_usable_addr(server_ip)
    client_ip, client_port = ip_addr.split(':')
    # Register client on server
    register_req = rpc.ClientRegisterRequest(ip_addr)
    for server_id in range(num_servers):
        rpc.send_request(server_id, register_req)
    # wait server connect back
    rpc.receiver_wait(client_ip, client_port, num_servers)
    # recv client ID from server
    res = rpc.recv_response()
    rpc.set_rank(res.client_id)
    print("Machine (%d) client (%d) connect to server successfuly!" \
        % (machine_id, rpc.get_rank()))
    # get total number of client
    get_client_num_req = rpc.GetNumberClientsRequest(rpc.get_rank())
    rpc.send_request(0, get_client_num_req)
    res = rpc.recv_response()
    rpc.set_num_client(res.num_client)
    from .dist_context import exit_client, set_initialized
    atexit.register(exit_client)
    set_initialized()

def shutdown_servers():
    """Issue commands to remote servers to shut them down.

    Raises
    ------
    ConnectionError : If anything wrong with the connection.
    """
    if rpc.get_rank() == 0:  # Only client_0 issue this command
        req = rpc.ShutDownRequest(rpc.get_rank())
        for server_id in range(rpc.get_num_server()):
            rpc.send_request(server_id, req)
