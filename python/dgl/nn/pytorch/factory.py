"""Modules that transforms between graphs and between graph and tensors."""
import torch.nn as nn
from ...transform import knn_graph, segmented_knn_graph

def pairwise_squared_distance(x):
    '''
    x : (n_samples, n_points, dims)
    return : (n_samples, n_points, n_points)
    '''
    x2s = (x * x).sum(-1, keepdim=True)
    return x2s + x2s.transpose(-1, -2) - 2 * x @ x.transpose(-1, -2)


class KNNGraph(nn.Module):
    r"""

    Description
    -----------
    Layer that transforms one point set into a graph, or a batch of
    point sets with the same number of points into a union of those graphs.

    The KNNGraph is implemented in the following steps:

    1. Compute an NxN matrix of pairwise distance for all points.
    2. Pick the k points with the smallest distance for each point as their k-nearest neighbors.
    3. Construct a graph with edges to each point as a node from its k-nearest neighbors.

    The overall computational complexity is :math:`O(N^2(logN + D)`.

    If a batch of point sets is provided, the point :math:`j` in point
    set :math:`i` is mapped to graph node ID: :math:`i \times M + j`, where
    :math:`M` is the number of nodes in each point set.

    The predecessors of each node are the k-nearest neighbors of the
    corresponding point.

    Parameters
    ----------
    k : int
        The number of neighbors.

    Notes
    -----
    The nearest neighbors found for a node include the node itself.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import torch
    >>> from dgl.nn.pytorch.factory import KNNGraph
    >>>
    >>> kg = KNNGraph(2)
    >>> x = torch.tensor([[0,1],
                          [1,2],
                          [1,3],
                          [100, 101],
                          [101, 102],
                          [50, 50]])
    >>> g = kg(x)
    >>> print(g.edges())
        (tensor([0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]),
         tensor([0, 0, 1, 2, 1, 2, 5, 3, 4, 3, 4, 5]))
    """
    def __init__(self, k):
        super(KNNGraph, self).__init__()
        self.k = k

    #pylint: disable=invalid-name
    def forward(self, x, algorithm='topk'):
        """

        Forward computation.

        Parameters
        ----------
        x : Tensor
            :math:`(M, D)` or :math:`(N, M, D)` where :math:`N` means the
            number of point sets, :math:`M` means the number of points in
            each point set, and :math:`D` means the size of features.
        algorithm : str, optional
            Algorithm used to compute the k-nearest neighbors.

            * 'topk' will use topk algorithm (quick-select or sorting,
            depending on backend implementation)
            * 'kd-tree' will use kd-tree algorithm (cpu version)

            (default: 'topk')

        Returns
        -------
        DGLGraph
            A DGLGraph without features.
        """
        return knn_graph(x, self.k, algorithm)


class SegmentedKNNGraph(nn.Module):
    r"""

    Description
    -----------
    Layer that transforms one point set into a graph, or a batch of
    point sets with different number of points into a union of those graphs.

    If a batch of point sets is provided, then the point :math:`j` in the point
    set :math:`i` is mapped to graph node ID:
    :math:`\sum_{p<i} |V_p| + j`, where :math:`|V_p|` means the number of
    points in the point set :math:`p`.

    The predecessors of each node are the k-nearest neighbors of the
    corresponding point.

    Parameters
    ----------
    k : int
        The number of neighbors.

    Notes
    -----
    The nearest neighbors found for a node include the node itself.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import torch
    >>> from dgl.nn.pytorch.factory import SegmentedKNNGraph
    >>>
    >>> kg = SegmentedKNNGraph(2)
    >>> x = torch.tensor([[0,1],
    ...                   [1,2],
    ...                   [1,3],
    ...                   [100, 101],
    ...                   [101, 102],
    ...                   [50, 50],
    ...                   [24,25],
    ...                   [25,24]])
    >>> g = kg(x, [3,3,2])
    >>> print(g.edges())
    (tensor([0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7]),
     tensor([0, 0, 1, 2, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 6, 7]))
    >>>

    """
    def __init__(self, k):
        super(SegmentedKNNGraph, self).__init__()
        self.k = k

    #pylint: disable=invalid-name
    def forward(self, x, segs, algorithm='topk'):
        r"""Forward computation.

        Parameters
        ----------
        x : Tensor
            :math:`(M, D)` where :math:`M` means the total number of points
            in all point sets, and :math:`D` means the size of features.
        segs : iterable of int
            :math:`(N)` integers where :math:`N` means the number of point
            sets.  The number of elements must sum up to :math:`M`. And any
            :math:`N` should :math:`\ge k`
        algorithm : str, optional
            Algorithm used to compute the k-nearest neighbors.

            * 'topk' will use topk algorithm (quick-select or sorting,
            depending on backend implementation)
            * 'kd-tree' will use kd-tree algorithm (cpu version)

            (default: 'topk')

        Returns
        -------
        DGLGraph
            A DGLGraph without features.
        """

        return segmented_knn_graph(x, self.k, segs, algorithm)
