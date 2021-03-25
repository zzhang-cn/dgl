import time

import dgl
import dgl.nn.pytorch as dglnn
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .. import utils


class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


def load_subtensor(g, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    batch_labels = g.ndata['labels'][seeds].to(device)
    return batch_inputs, batch_labels


@utils.benchmark('time', 600)
@utils.parametrize('data', ['reddit', 'ogbn-products'])
def track_time(data):
    dataset = utils.process_data(data)
    device = utils.get_bench_device()
    g = dataset[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    in_feats = g.ndata['features'].shape[1]
    n_classes = dataset.num_classes

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    g.create_formats_()

    num_runs = 3
    num_hidden = 16
    num_layers = 2
    fan_out = '10,25'
    batch_size = 1024
    lr = 0.003
    dropout = 0.5
    num_workers = 4

    train_nid = th.nonzero(g.ndata['train_mask'], as_tuple=True)[0]

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers)

    epoch_times = []

    for run in range(num_runs):
        # Define model and optimizer
        model = SAGE(in_feats, num_hidden, n_classes,
                     num_layers, F.relu, dropout)
        model = model.to(device)
        loss_fcn = nn.CrossEntropyLoss()
        loss_fcn = loss_fcn.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # dry run
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            #batch_inputs, batch_labels = load_subtensor(g, seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step >= 4:
                break

        # Training loop
        avg = 0
        iter_tput = []

        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            t0 = time.time()

            # Load the input features as well as output labels
            #batch_inputs, batch_labels = load_subtensor(g, seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t1 = time.time()

            epoch_times.append(t1 - t0)

            if step >= 9:  # time 10 loops
                break

    avg_epoch_time = np.mean(epoch_times)
    std_epoch_time = np.std(epoch_times)

    std_const = 1.5
    low_boundary = avg_epoch_time - std_epoch_time * std_const
    high_boundary = avg_epoch_time + std_epoch_time * std_const

    valid_epoch_times = np.array(epoch_times)[(
        epoch_times >= low_boundary) & (epoch_times <= high_boundary)]
    avg_valid_epoch_time = np.mean(valid_epoch_times)

    # TODO: delete logging for final version
    print(f'Number of epoch times: {len(epoch_times)}')
    print(f'Number of valid epoch times: {len(valid_epoch_times)}')
    print(f'Avg epoch times: {avg_epoch_time}')
    print(f'Avg valid epoch times: {avg_valid_epoch_time}')

    return avg_valid_epoch_time
