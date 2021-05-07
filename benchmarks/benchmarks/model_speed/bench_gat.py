import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv

from .. import utils


class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits


@utils.benchmark('time')
@utils.parametrize('data', ['cora', 'pubmed'])
def track_time(data):
    dataset = utils.process_data(data)
    device = utils.get_bench_device()

    g = dataset[0].to(device)

    num_runs = 3
    num_epochs = 200

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    in_feats = features.shape[1]
    n_classes = dataset.num_classes

    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    # create model
    model = GAT(1, in_feats, 8, n_classes, [8, 1], F.elu,
                0.6, 0.6, 0.2, False)
    loss_fcn = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    model.train()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-2,
                                 weight_decay=5e-4)

    timer = utils.ModelSpeedTimer()

    for run in range(num_runs):
        # dry run
        for epoch in range(10):
            logits = model(g, features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # timing
        for epoch in range(num_epochs):
            with timer as t:
                logits = model(g, features)
                loss = loss_fcn(logits[train_mask], labels[train_mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return timer.average_epoch_time
