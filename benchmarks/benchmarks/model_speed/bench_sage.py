import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv

from .. import utils


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes,
                           aggregator_type))  # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


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
    model = GraphSAGE(in_feats, 16, n_classes, 1, F.relu, 0.5, 'gcn')
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
