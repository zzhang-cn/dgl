import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import utils


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, graph, feat):
        with graph.local_scope():
            graph.ndata['ci'] = torch.pow(
                graph.out_degrees().float().clamp(min=1), -0.5)
            graph.ndata['cj'] = torch.pow(
                graph.in_degrees().float().clamp(min=1), -0.5)
            graph.ndata['h'] = feat
            graph.update_all(self.mfunc, self.rfunc)
            h = graph.ndata['h']
            h = torch.matmul(h, self.weight) + self.bias
            if self.activation is not None:
                h = self.activation(h)
            return h

    def mfunc(self, edges):
        return {'m': edges.src['h'], 'ci': edges.src['ci']}

    def rfunc(self, nodes):
        ci = nodes.mailbox['ci'].unsqueeze(2)
        newh = (nodes.mailbox['m'] * ci).sum(1) * nodes.data['cj'].unsqueeze(1)
        return {'h': newh}


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


@utils.benchmark('time', timeout=300)
@utils.parametrize('data', ['cora', 'pubmed'])
def track_time(data):
    dataset = utils.process_data(data)
    device = utils.get_bench_device()

    g = dataset[0].to(device).int()

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

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(in_feats, 16, n_classes, 1, F.relu, 0.5)
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
        for epoch in range(5):
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
