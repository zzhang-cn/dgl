import os
import copy

import numpy as np
import torch
import dgl
import networkx as nx
from torch.utils.data import Dataset, DataLoader


def build_dense_graph(n_particles):
    g = nx.complete_graph(n_particles)
    return dgl.from_networkx(g)


class MultiBodyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.zipfile = np.load(self.path)
        self.node_state = self.zipfile['data']
        self.node_label = self.zipfile['label']
        self.n_particles = self.zipfile['n_particles']

    def __len__(self):
        return self.node_state.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        node_state = self.node_state[idx, :, :]
        node_label = self.node_label[idx, :, :]
        return (node_state, node_label)


class MultiBodyTrainDataset(MultiBodyDataset):
    def __init__(self, data_path='./data/'):
        super(MultiBodyTrainDataset, self).__init__(
            data_path+'n_body_train.npz')
        self.stat_median = self.zipfile['median']
        self.stat_max = self.zipfile['max']
        self.stat_min = self.zipfile['min']


class MultiBodyValidDataset(MultiBodyDataset):
    def __init__(self, data_path='./data/'):
        super(MultiBodyValidDataset, self).__init__(
            data_path+'n_body_valid.npz')


class MultiBodyTestDataset(MultiBodyDataset):
    def __init__(self, data_path='./data/'):
        super(MultiBodyTestDataset, self).__init__(data_path+'n_body_test.npz')
        self.test_traj = self.zipfile['test_traj']
        self.first_frame = torch.from_numpy(self.zipfile['first_frame'])

# Construct fully connected graph


class MultiBodyGraphCollator:
    def __init__(self, n_particles):
        self.n_particles = n_particles
        self.graph = dgl.from_networkx(nx.complete_graph(self.n_particles))

    def __call__(self, batch):
        graph_list = []
        data_list = []
        label_list = []
        for frame in batch:
            graph_list.append(copy.deepcopy(self.graph))
            data_list.append(torch.from_numpy(frame[0]))
            label_list.append(torch.from_numpy(frame[1]))

        graph_batch = dgl.batch(graph_list)
        data_batch = torch.vstack(data_list)
        label_batch = torch.vstack(label_list)
        return graph_batch, data_batch, label_batch
