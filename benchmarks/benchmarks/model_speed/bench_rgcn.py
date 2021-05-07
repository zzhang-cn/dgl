import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv

from .. import utils


class RGCN(nn.Module):
    def __init__(self,
                 num_nodes,
                 n_hidden,
                 num_classes,
                 num_rels,
                 num_bases,
                 num_hidden_layers,
                 dropout,
                 lowmem):
        super(RGCN, self).__init__()
        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConv(num_nodes, n_hidden, num_rels, "basis",
                                        num_bases, activation=F.relu, dropout=dropout,
                                        low_mem=lowmem))
        # h2h
        for i in range(num_hidden_layers):
            self.layers.append(RelGraphConv(n_hidden, n_hidden, num_rels, "basis",
                                            num_bases, activation=F.relu, dropout=dropout,
                                            low_mem=lowmem))
        # o2h
        self.layers.append(RelGraphConv(n_hidden, num_classes, num_rels, "basis",
                                        num_bases, activation=None, low_mem=lowmem))

    def forward(self, g, h, r, norm):
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h


@utils.benchmark('time', 300)
@utils.parametrize('data', ['aifb'])
@utils.parametrize('lowmem', [True, False])
@utils.parametrize('use_type_count', [True, False])
def track_time(data, lowmem, use_type_count):
    # args
    if data == 'aifb':
        num_bases = -1
        l2norm = 0.
    elif data == 'am':
        num_bases = 40
        l2norm = 5e-4
    else:
        raise ValueError()

    dataset = utils.process_data(data)
    device = utils.get_bench_device()

    g = dataset[0]

    num_runs = 3
    num_epochs = 30

    num_rels = len(g.canonical_etypes)
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = g.nodes[category].data.pop('train_mask').bool().to(device)
    test_mask = g.nodes[category].data.pop('test_mask').bool().to(device)
    labels = g.nodes[category].data.pop('labels').to(device)

    # calculate norm for each edge type and store in edge
    for canonical_etype in g.canonical_etypes:
        u, v, eid = g.all_edges(form='all', etype=canonical_etype)
        _, inverse_index, count = torch.unique(
            v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = 1. / degrees.float()
        norm = norm.unsqueeze(1)
        g.edges[canonical_etype].data['norm'] = norm

    # get target category id
    category_id = len(g.ntypes)
    for i, ntype in enumerate(g.ntypes):
        if ntype == category:
            category_id = i

    if use_type_count:
        g, _, edge_type = dgl.to_homogeneous(
            g, edata=['norm'], return_count=True)
        g = g.to(device)
    else:
        g = dgl.to_homogeneous(g, edata=['norm']).to(device)
        edge_type = g.edata.pop(dgl.ETYPE).long()

    num_nodes = g.number_of_nodes()
    edge_norm = g.edata['norm']

    # find out the target node ids in g
    target_idx = torch.where(g.ndata[dgl.NTYPE] == category_id)[0]
    train_idx = target_idx[train_mask]
    test_idx = target_idx[test_mask]
    train_labels = labels[train_mask]
    test_labels = labels[test_mask]

    # since the nodes are featureless, the input feature is then the node id.
    feats = torch.arange(num_nodes, device=device)

    # create model
    model = RGCN(num_nodes,
                 16,
                 num_classes,
                 num_rels,
                 num_bases,
                 0,
                 0,
                 lowmem).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-2,
                                 weight_decay=l2norm)

    model.train()

    timer = utils.ModelSpeedTimer()

    for run in range(num_runs):
        # dry run
        for epoch in range(10):
            logits = model(g, feats, edge_type, edge_norm)
            loss = F.cross_entropy(logits[train_idx], train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # timing
        for epoch in range(num_epochs):
            with timer as t:
                logits = model(g, feats, edge_type, edge_norm)
                loss = F.cross_entropy(logits[train_idx], train_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return timer.average_epoch_time
