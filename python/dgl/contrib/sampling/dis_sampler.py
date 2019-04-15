# This file contains DGL distributed samplers APIs.
from ...network import _send_nodeflow, _recv_nodeflow
from ...network import _create_sender, _create_receiver
from ...network import _finalize_sender, _finalize_receiver
from ...network import _add_receiver_addr, _sender_connect, _receiver_wait

from multiprocessing import Pool
from abc import ABCMeta, abstractmethod

class SamplerPool(object):
    """SamplerPool is an abstract class, in which the worker method 
    should be implemented by users. SamplerPool will fork() N (N = num_worker)
    child processes, and each process will perform worker() method independently.
    Note that, the fork() API will use shared memory for N process and the OS will
    perfrom copy-on-write only when developers write that piece of memory.

    Users can use this class like this:

      class MySamplerPool(SamplerPool):

          def worker(self):
              # Do anything here #

      if __name__ == '__main__':
          ...
          args = parser.parse_args()
          pool = MySamplerPool()
          pool.start(args.num_sender, args)
    """
    __metaclass__ = ABCMeta

    def start(self, num_worker, args):
        """Start sampler pool

        Parameters
        ----------
        num_worker : int
            number of worker (number of child process)
        args : arguments
            arguments passed by user
        """
        p = Pool()
        for i in range(num_worker):
            print("Start child process %d ..." % i)
            p.apply_async(self.worker, args=(args,))
        # Waiting for all subprocesses done ...
        p.close()
        p.join()

    @abstractmethod
    def worker(self, args):
        pass

class SamplerSender(object):
    """SamplerSender for DGL distributed training.

    Users use SamplerSender to send sampled subgraph (NodeFlow) 
    to remote SamplerReceiver. Note that a SamplerSender can connect 
    to multiple SamplerReceiver.

    Parameters
    ----------
    recv_addr : dict
        address dict of SamplerReceiver, where
        key is recv_id and value is recv address, e.g.,

        { 0:'168.12.23.45:50051', 
          1:'168.12.23.21:50051', 
          2:'168.12.46.12:50051' }

    """
    def __init__(self, namebook):
        assert len(namebook) > 0, 'recv_addr cannot be empty.'
        self._namebook = namebook
        self._sender = _create_sender()
        for ID, addr in self._namebook.items():
            vec = addr.split(':')
            _add_receiver_addr(self._sender, vec[0], int(vec[1]), ID)
        _sender_connect(self._sender)

    def __del__(self):
        """Finalize Sender
        """
        _finalize_sender(self._sender)

    def send(self, nodeflow, recv_id):
        """Send sampled subgraph (NodeFlow) to remote trainer.

        Parameters
        ----------
        nodeflow : NodeFlow
            sampled NodeFlow object
        recv_id : int
            receiver ID
        """
        _send_nodeflow(self._sender, nodeflow, recv_id)

class SamplerReceiver(object):
    """SamplerReceiver for DGL distributed training.

    Users use SamplerReceiver to receive sampled subgraph (NodeFlow) 
    from remote SamplerSender. Note that SamplerReceiver can receive messages 
    from multiple SamplerSenders concurrently, by given the num_sender parameter. 
    Note that, only when all SamplerSenders connect to SamplerReceiver, it can 
    start its job.

    Parameters
    ----------
    addr : str
        address of SamplerReceiver, e.g., '127.0.0.1:50051'
    num_sender : int
        total number of SamplerSender
    """
    def __init__(self, addr, num_sender):
        self._addr = addr
        self._num_sender = num_sender
        self._receiver = _create_receiver()
        vec = self._addr.split(':')
        _receiver_wait(self._receiver, vec[0], int(vec[1]), self._num_sender);

    def __del__(self):
        """Finalize Receiver
        """
        _finalize_receiver(self._receiver)

    def recv(self, graph):
        """Receive a NodeFlow object from remote sampler.

        Parameters
        ----------
        graph : DGLGraph
            The parent graph

        Returns
        -------
        NodeFlow
            received NodeFlow object
        """
        return _recv_nodeflow(self._receiver, graph)
