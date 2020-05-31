"""Define graph partition book."""

import numpy as np

from .. import backend as F
from .. import utils

class GraphPartitionBook:
    """GraphPartitionBook is used to store parition information.

    Parameters
    ----------
    part_id : int
        partition id of current GraphPartitionBook
    num_parts : int
        number of total partitions
    node_map : tensor
        global node id mapping to partition id
    edge_map : tensor
        global edge id mapping to partition id
    global_nid : tensor
        global node id
    global_eid : tensor
        global edge id
    """
    def __init__(self, part_id, num_parts, node_map, edge_map, global_nid, global_eid):
        assert part_id >= 0, 'part_id cannot be a negative number.'
        assert num_parts > 0, 'num_parts must be greater than zero.'
        self._part_id = part_id
        self._num_partitions = num_parts
        node_map = utils.toindex(node_map)
        self._nid2partid = node_map.tousertensor()
        edge_map = utils.toindex(edge_map)
        self._eid2partid = edge_map.tousertensor()
        # Get meta data of GraphPartitionBook
        self._partition_meta_data = []
        _, nid_count = np.unique(F.asnumpy(self._nid2partid), return_counts=True)
        _, eid_count = np.unique(F.asnumpy(self._eid2partid), return_counts=True)
        for partid in range(self._num_partitions):
            part_info = {}
            part_info['machine_id'] = partid
            part_info['num_nodes'] = nid_count[partid]
            part_info['num_edges'] = eid_count[partid]
            self._partition_meta_data.append(part_info)
        # Get partid2nids
        self._partid2nids = []
        sorted_nid = F.tensor(np.argsort(F.asnumpy(self._nid2partid)))
        start = 0
        for offset in nid_count:
            part_nids = sorted_nid[start:start+offset]
            start += offset
            self._partid2nids.append(part_nids)
        # Get partid2eids
        self._partid2eids = []
        sorted_eid = F.tensor(np.argsort(F.asnumpy(self._eid2partid)))
        start = 0
        for offset in eid_count:
            part_eids = sorted_eid[start:start+offset]
            start += offset
            self._partid2eids.append(part_eids)
        # Get nidg2l
        self._nidg2l = [None] * self._num_partitions
        global_id = global_nid
        max_global_id = np.amax(F.asnumpy(global_id))
        # TODO(chao): support int32 index
        g2l = F.zeros((max_global_id+1), F.int64, F.context(global_id))
        g2l = F.scatter_row(g2l, global_id, F.arange(0, len(global_id)))
        self._nidg2l[self._part_id] = g2l
        # Get eidg2l
        self._eidg2l = [None] * self._num_partitions
        global_id = global_eid
        max_global_id = np.amax(F.asnumpy(global_id))
        # TODO(chao): support int32 index
        g2l = F.zeros((max_global_id+1), F.int64, F.context(global_id))
        g2l = F.scatter_row(g2l, global_id, F.arange(0, len(global_id)))
        self._eidg2l[self._part_id] = g2l


    def num_partitions(self):
        """Return the number of partitions.

        Returns
        -------
        int
            number of partitions
        """
        return self._num_partitions


    def metadata(self):
        """Return the partition meta data.

        The meta data includes:

        * The machine ID.
        * The machine IP address.
        * Number of nodes and edges of each partition.

        Examples
        --------
        >>> print(g.get_partition_book().metadata())
        >>> [{'machine_id' : 0, 'num_nodes' : 3000, 'num_edges' : 5000},
        ...  {'machine_id' : 1, 'num_nodes' : 2000, 'num_edges' : 4888},
        ...  ...]

        Returns
        -------
        list[dict[str, any]]
            Meta data of each partition.
        """
        return self._partition_meta_data


    def nid2partid(self, nids):
        """From global node IDs to partition IDs

        Parameters
        ----------
        nids : tensor
            global node IDs

        Returns
        -------
        tensor
            partition IDs
        """
        return F.gather_row(self._nid2partid, nids)


    def eid2partid(self, eids):
        """From global edge IDs to partition IDs

        Parameters
        ----------
        eids : tensor
            global edge IDs

        Returns
        -------
        tensor
            partition IDs
        """
        return F.gather_row(self._eid2partid, eids)


    def partid2nids(self, partid):
        """From partition id to node IDs

        Parameters
        ----------
        partid : int
            partition id

        Returns
        -------
        tensor
            node IDs
        """
        return self._partid2nids[partid]


    def partid2eids(self, partid):
        """From partition id to edge IDs

        Parameters
        ----------
        partid : int
            partition id

        Returns
        -------
        tensor
            edge IDs
        """
        return self._partid2eids[partid]


    def nid2localnid(self, nids, partid):
        """Get local node IDs within the given partition.

        Parameters
        ----------
        nids : tensor
            global node IDs
        partid : int
            partition ID

        Returns
        -------
        tensor
             local node IDs
        """
        if partid != self._part_id:
            raise RuntimeError('Now GraphPartitionBook does not support \
                getting remote tensor of nid2localnid.')
        return F.gather_row(self._nidg2l[partid], nids)


    def eid2localeid(self, eids, partid):
        """Get the local edge ids within the given partition.

        Parameters
        ----------
        eids : tensor
            global edge ids
        partid : int
            partition ID

        Returns
        -------
        tensor
             local edge ids
        """
        if partid != self._part_id:
            raise RuntimeError('Now GraphPartitionBook does not support \
                getting remote tensor of eid2localeid.')
        return F.gather_row(self._eidg2l[partid], eids)


    def get_partition(self, partid):
        """Get the graph of one partition.

        Parameters
        ----------
        partid : int
            Partition ID.

        Returns
        -------
        DGLGraph
            The graph of the partition.
        """
        if partid != self._part_id:
            raise RuntimeError('Now GraphPartitionBook does not support \
                getting remote partitions.')

        return self._graph

class PartitionPolicy(object):
    """Wrapper for GraphPartitionBook and RangePartitionBook. 

    We can extend this class to support HeteroGraph in the future.

    Parameters
    ----------
    policy_str : str
        partition-policy string, e.g., 'edge' or 'node'.
    part_id : int
        partition ID
    partition_book : GraphPartitionBook or RangePartitionBook
        Main class storing the partition information
    """
    def __init__(self, policy_str, part_id, partition_book):
        # TODO(chao): support more policies for HeteroGraph
        assert policy_str in ('edge', 'node'), 'policy_str must be \'edge\' or \'node\'.'
        assert part_id >= 0, 'part_id %d cannot be a negative number.' % part_id
        self._policy_str = policy_str
        self._part_id = part_id
        self._partition_book = partition_book

    @property
    def policy_str(self):
        return self._policy_str

    @property
    def part_id(self):
        return self._part_id

    @property
    def partition_book(self):
        return self._partition_book
    
    def to_local(self, id_tensor):
        """Mapping global ID to local ID.

        Parameters
        ----------
        id_tensor : tensor
            Gloabl ID tensor

        Return
        ------
        tensor
            local ID tensor
        """
        if self._policy_str == 'edge':
            return self._partition_book.eid2localeid(id_tensor, self._part_id)
        elif self._policy_str == 'node':
            return self._partition_book.nid2localnid(id_tensor, self._part_id)
        else:
            raise RuntimeError('Cannot support policy: %s ' % self._policy_str)

    def to_partid(self, id_tensor):
        """Mapping global ID to partition ID.

        Parameters
        ----------
        id_tensor : tensor
            Global ID tensor

        Return
        ------
        tensor
            partition ID
        """
        if self._policy_str == 'edge':
            return self._partition_book.eid2partid(id_tensor)
        elif self._policy_str == 'node':
            return self._partition_book.nid2partid(id_tensor)
        else:
            raise RuntimeError('Cannot support policy: %s ' % self._policy_str)

    def get_data_size(self):
        """Get data size of current partition.

        Returns
        -------
        int
            data size
        """
        if self._policy_str == 'edge':
            return len(self._partition_book.partid2eids(self._part_id))
        elif self._policy_str == 'node':
            return len(self._partition_book.partid2nids(self._part_id))
        else:
            raise RuntimeError('Cannot support policy: %s ' % self._policy_str)

    # TODO(chao): more APIs?