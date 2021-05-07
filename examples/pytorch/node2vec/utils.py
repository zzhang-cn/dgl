import argparse
from dgl.data import CitationGraphDataset
from ogb.nodeproppred import *
from ogb.linkproppred import *


def load_graph(name):
    cite_graphs = ['cora', 'citeseer', 'pubmed']

    if name in cite_graphs:
        dataset = CitationGraphDataset(name)
        graph = dataset[0]

        nodes = graph.nodes()
        y = graph.ndata['label']
        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['test_mask']

        nodes_train, y_train = nodes[train_mask], y[train_mask]
        nodes_val, y_val = nodes[val_mask], y[val_mask]
        eval_set = [(nodes_train, y_train), (nodes_val, y_val)]

    elif name.startswith('ogbn'):

        dataset = DglNodePropPredDataset(name)
        graph, y = dataset[0]
        split_nodes = dataset.get_idx_split()
        nodes = graph.nodes()

        train_idx = split_nodes['train']
        val_idx = split_nodes['valid']

        nodes_train, y_train = nodes[train_idx], y[train_idx]
        nodes_val, y_val = nodes[val_idx], y[val_idx]
        eval_set = [(nodes_train, y_train), (nodes_val, y_val)]

    elif name.startswith('ogbl'):
        # link prediction graph dataset can not be evaluated.
        dataset = DglLinkPropPredDataset(name)
        eval_set = {}
        graph = dataset[0]

    else:
        raise ValueError("Dataset name error!")

    return graph, eval_set


def parse_arguments():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description='Node2vec')
    parser.add_argument('--dataset', type=str, default='cora')
    # 'train' for training node2vec model, 'test' for testing speed of random walk
    parser.add_argument('--task', type=str, default='train')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--walk_length', type=int, default=50)
    parser.add_argument('--p', type=float, default=0.25)
    parser.add_argument('--q', type=float, default=4.0)
    parser.add_argument('--num_walks', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()

    return args
