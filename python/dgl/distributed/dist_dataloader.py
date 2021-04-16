# pylint: disable=global-variable-undefined, invalid-name
"""Multiprocess dataloader for distributed training"""
import multiprocessing as mp
from queue import Queue
import traceback

from .dist_context import get_sampler_pool
from .. import backend as F

__all__ = ["DistDataLoader"]


def call_collate_fn(name, next_data):
    """Call collate function"""
    try:
        result = DGL_GLOBAL_COLLATE_FNS[name](next_data)
        DGL_GLOBAL_MP_QUEUES[name].put(result)
    except Exception as e:
        traceback.print_exc()
        print(e)
        raise e
    return 1

DGL_GLOBAL_COLLATE_FNS = {}
DGL_GLOBAL_MP_QUEUES = {}

def init_fn(barrier, name, collate_fn, queue):
    """Initialize setting collate function and mp.Queue in the subprocess"""
    global DGL_GLOBAL_COLLATE_FNS
    global DGL_GLOBAL_MP_QUEUES
    DGL_GLOBAL_MP_QUEUES[name] = queue
    DGL_GLOBAL_COLLATE_FNS[name] = collate_fn
    barrier.wait()
    return 1

def cleanup_fn(barrier, name):
    """Clean up the data of a dataloader in the worker process"""
    global DGL_GLOBAL_COLLATE_FNS
    global DGL_GLOBAL_MP_QUEUES
    del DGL_GLOBAL_MP_QUEUES[name]
    del DGL_GLOBAL_COLLATE_FNS[name]
    # sleep here is to ensure this function is executed in all worker processes
    # probably need better solution in the future
    barrier.wait()
    return 1


def enable_mp_debug():
    """Print multiprocessing debug information. This is only
    for debug usage"""
    import logging
    logger = mp.log_to_stderr()
    logger.setLevel(logging.DEBUG)

DATALOADER_ID = 0

class DistDataLoader:
    """DGL customized multiprocessing dataloader.

    DistDataLoader provides a similar interface to Pytorch's DataLoader to generate mini-batches
    with multiprocessing. It utilizes the worker processes created by
    :func:`dgl.distributed.initialize` to parallelize sampling.

    Parameters
    ----------
    dataset: a tensor
        A tensor of node IDs or edge IDs.
    batch_size: int
        The number of samples per batch to load.
    shuffle: bool, optional
        Set to ``True`` to have the data reshuffled at every epoch (default: ``False``).
    collate_fn: callable, optional
        The function is typically used to sample neighbors of the nodes in a batch
        or the endpoint nodes of the edges in a batch.
    drop_last: bool, optional
        Set to ``True`` to drop the last incomplete batch, if the dataset size is not
        divisible by the batch size. If ``False`` and the size of dataset is not divisible
        by the batch size, then the last batch will be smaller. (default: ``False``)
    queue_size: int, optional
        Size of multiprocessing queue

    Examples
    --------
    >>> g = dgl.distributed.DistGraph('graph-name')
    >>> def sample(seeds):
    ...     seeds = th.LongTensor(np.asarray(seeds))
    ...     frontier = dgl.distributed.sample_neighbors(g, seeds, 10)
    ...     return dgl.to_block(frontier, seeds)
    >>> dataloader = dgl.distributed.DistDataLoader(dataset=nodes, batch_size=1000,
                                                    collate_fn=sample, shuffle=True)
    >>> for block in dataloader:
    ...     feat = g.ndata['features'][block.srcdata[dgl.NID]]
    ...     labels = g.ndata['labels'][block.dstdata[dgl.NID]]
    ...     pred = model(block, feat)

    Note
    ----
    When performing DGL's distributed sampling with multiprocessing, users have to use this class
    instead of Pytorch's DataLoader because DGL's RPC requires that all processes establish
    connections with servers before invoking any DGL's distributed API. Therefore, this dataloader
    uses the worker processes created in :func:`dgl.distributed.initialize`.

    Note
    ----
    This dataloader does not guarantee the iteration order. For example,
    if dataset = [1, 2, 3, 4], batch_size = 2 and shuffle = False, the order of [1, 2]
    and [3, 4] is not guaranteed.
    """

    def __init__(self, dataset, batch_size, shuffle=False, collate_fn=None, drop_last=False,
                 queue_size=None):
        self.pool, self.num_workers = get_sampler_pool()
        if queue_size is None:
            queue_size = self.num_workers * 4 if self.num_workers > 0 else 4
        self.queue_size = queue_size
        self.batch_size = batch_size
        self.num_pending = 0
        self.collate_fn = collate_fn
        self.current_pos = 0
        if self.pool is not None:
            m = mp.Manager()
            self.queue = m.Queue(maxsize=queue_size)
        else:
            self.queue = Queue(maxsize=queue_size)
        self.drop_last = drop_last
        self.recv_idxs = 0
        self.shuffle = shuffle
        self.is_closed = False

        self.dataset = F.tensor(dataset)
        self.expected_idxs = len(dataset) // self.batch_size
        if not self.drop_last and len(dataset) % self.batch_size != 0:
            self.expected_idxs += 1

        # We need to have a unique ID for each data loader to identify itself
        # in the sampler processes.
        global DATALOADER_ID
        self.name = "dataloader-" + str(DATALOADER_ID)
        DATALOADER_ID += 1

        if self.pool is not None:
            results = []
            barrier = m.Barrier(self.num_workers)
            for _ in range(self.num_workers):
                results.append(self.pool.apply_async(
                    init_fn, args=(barrier, self.name, self.collate_fn, self.queue)))
            for res in results:
                res.get()

    def __del__(self):
        # When the process exits, the process pool may have been closed. We should try
        # and get the process pool again and see if we need to clean up the process pool.
        self.pool, self.num_workers = get_sampler_pool()
        if self.pool is not None:
            results = []
            # Here we need to create the manager and barrier again.
            m = mp.Manager()
            barrier = m.Barrier(self.num_workers)
            for _ in range(self.num_workers):
                results.append(self.pool.apply_async(cleanup_fn, args=(barrier, self.name,)))
            for res in results:
                res.get()

    def __next__(self):
        num_reqs = self.queue_size - self.num_pending
        for _ in range(num_reqs):
            self._request_next_batch()
        if self.recv_idxs < self.expected_idxs:
            result = self.queue.get(timeout=1800)
            self.recv_idxs += 1
            self.num_pending -= 1
            return result
        else:
            assert self.num_pending == 0
            raise StopIteration

    def __iter__(self):
        if self.shuffle:
            self.dataset = F.rand_shuffle(self.dataset)
        self.recv_idxs = 0
        self.current_pos = 0
        self.num_pending = 0
        return self

    def _request_next_batch(self):
        next_data = self._next_data()
        if next_data is None:
            return
        elif self.pool is not None:
            self.pool.apply_async(call_collate_fn, args=(self.name, next_data, ))
        else:
            result = self.collate_fn(next_data)
            self.queue.put(result)
        self.num_pending += 1

    def _next_data(self):
        if self.current_pos == len(self.dataset):
            return None

        end_pos = 0
        if self.current_pos + self.batch_size > len(self.dataset):
            if self.drop_last:
                return None
            else:
                end_pos = len(self.dataset)
        else:
            end_pos = self.current_pos + self.batch_size
        ret = self.dataset[self.current_pos:end_pos]
        self.current_pos = end_pos
        return ret
