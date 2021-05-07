import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .. import utils


class NegativeSampler(object):
    def __init__(self, g, k, neg_share=False):
        self.weights = g.in_degrees().float() ** 0.75
        self.k = k
        self.neg_share = neg_share

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)
        n = len(src)
        if self.neg_share and n % self.k == 0:
            dst = self.weights.multinomial(n, replacement=True)
            dst = dst.view(-1, 1, self.k).expand(-1, self.k, -1).flatten()
        else:
            dst = self.weights.multinomial(n*self.k, replacement=True)
        src = src.repeat_interleave(self.k)
        return src, dst


def load_subtensor(g, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    return batch_inputs


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


def load_subtensor(g, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    return batch_inputs


class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        score = th.cat([pos_score, neg_score])
        label = th.cat(
            [th.ones_like(pos_score), th.zeros_like(neg_score)]).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss


@utils.benchmark('time', 600)
@utils.parametrize('data', ['reddit'])
@utils.parametrize('num_negs', [2, 8, 32])
@utils.parametrize('batch_size', [1024, 2048, 8192])
def track_time(data, num_negs, batch_size):
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
    lr = 0.003
    dropout = 0.5
    num_workers = 4
    num_negs = 2

    n_edges = g.number_of_edges()
    train_seeds = np.arange(n_edges)

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in fan_out.split(',')])
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_seeds, sampler, exclude='reverse_id',
        # For each edge with ID e in Reddit dataset, the reverse edge is e ± |E|/2.
        reverse_eids=th.cat([
            th.arange(n_edges // 2, n_edges),
            th.arange(0, n_edges // 2)]),
        negative_sampler=NegativeSampler(g, num_negs),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers)

    # Define model and optimizer
    model = SAGE(in_feats, num_hidden, n_classes,
                 num_layers, F.relu, dropout)
    model = model.to(device)
    loss_fcn = CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    timer = utils.ModelSpeedTimer()

    for run in range(num_runs):
        # dry run
        for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            batch_inputs = load_subtensor(g, input_nodes, device)

            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            blocks = [block.int().to(device) for block in blocks]
            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, pos_graph, neg_graph)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step >= 4:
                break

        # Training loop
        avg = 0
        iter_tput = []

        for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
            with timer as t:
                # Load the input features as well as output labels
                batch_inputs = load_subtensor(g, input_nodes, device)

                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)
                blocks = [block.int().to(device) for block in blocks]
                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, pos_graph, neg_graph)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step >= 9:  # time 10 loops
                break

    return timer.average_epoch_time
