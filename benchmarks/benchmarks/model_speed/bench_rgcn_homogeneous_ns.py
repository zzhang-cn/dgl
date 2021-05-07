import itertools

import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import RelGraphConv

from .. import utils


class EntityClassify(nn.Module):
    """ Entity classification class for RGCN
    Parameters
    ----------
    device : int
        Device to run the layer.
    num_nodes : int
        Number of nodes.
    h_dim : int
        Hidden dim size.
    out_dim : int
        Output dim size.
    num_rels : int
        Numer of relation types.
    num_bases : int
        Number of bases. If is none, use number of relations.
    num_hidden_layers : int
        Number of hidden RelGraphConv Layer
    dropout : float
        Dropout
    use_self_loop : bool
        Use self loop if True, default False.
    low_mem : bool
        True to use low memory implementation of relation message passing function
        trade speed with memory consumption
    """

    def __init__(self,
                 device,
                 num_nodes,
                 h_dim,
                 out_dim,
                 num_rels,
                 num_bases=None,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False,
                 low_mem=False,
                 layer_norm=False):
        super(EntityClassify, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.low_mem = low_mem
        self.layer_norm = layer_norm

        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConv(
            self.h_dim, self.h_dim, self.num_rels, "basis",
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            low_mem=self.low_mem, dropout=self.dropout, layer_norm=layer_norm))
        # h2h
        for idx in range(self.num_hidden_layers):
            self.layers.append(RelGraphConv(
                self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                low_mem=self.low_mem, dropout=self.dropout, layer_norm=layer_norm))
        # h2o
        self.layers.append(RelGraphConv(
            self.h_dim, self.out_dim, self.num_rels, "basis",
            self.num_bases, activation=None,
            self_loop=self.use_self_loop,
            low_mem=self.low_mem, layer_norm=layer_norm))

    def forward(self, blocks, feats, norm=None):
        if blocks is None:
            # full graph training
            blocks = [self.g] * len(self.layers)
        h = feats
        for layer, block in zip(self.layers, blocks):
            block = block.to(self.device)
            h = layer(block, h, block.edata['etype'], block.edata['norm'])
        return h


class RelGraphEmbedLayer(nn.Module):
    r"""Embedding layer for featureless heterograph.
    Parameters
    ----------
    device : int
        Device to run the layer.
    num_nodes : int
        Number of nodes.
    node_tides : tensor
        Storing the node type id for each node starting from 0
    num_of_ntype : int
        Number of node types
    input_size : list of int
        A list of input feature size for each node type. If None, we then
        treat certain input feature as an one-hot encoding feature.
    embed_size : int
        Output embed size
    embed_name : str, optional
        Embed name
    """

    def __init__(self,
                 device,
                 num_nodes,
                 node_tids,
                 num_of_ntype,
                 input_size,
                 embed_size,
                 sparse_emb=False,
                 embed_name='embed'):
        super(RelGraphEmbedLayer, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.num_nodes = num_nodes
        self.sparse_emb = sparse_emb

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        self.num_of_ntype = num_of_ntype
        self.idmap = th.empty(num_nodes).long()

        for ntype in range(num_of_ntype):
            if input_size[ntype] is not None:
                input_emb_size = input_size[ntype].shape[1]
                embed = nn.Parameter(
                    th.Tensor(input_emb_size, self.embed_size))
                nn.init.xavier_uniform_(embed)
                self.embeds[str(ntype)] = embed

        self.node_embeds = th.nn.Embedding(
            node_tids.shape[0], self.embed_size, sparse=self.sparse_emb)
        nn.init.uniform_(self.node_embeds.weight, -1.0, 1.0)

    def forward(self, node_ids, node_tids, type_ids, features):
        """Forward computation
        Parameters
        ----------
        node_ids : tensor
            node ids to generate embedding for.
        node_tids : tensor
            node type ids
        features : list of features
            list of initial features for nodes belong to different node type.
            If None, the corresponding features is an one-hot encoding feature,
            else use the features directly as input feature and matmul a
            projection matrix.
        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """
        tsd_ids = node_ids.to(self.node_embeds.weight.device)
        embeds = th.empty(node_ids.shape[0],
                          self.embed_size, device=self.device)
        for ntype in range(self.num_of_ntype):
            if features[ntype] is not None:
                loc = node_tids == ntype
                embeds[loc] = features[ntype][type_ids[loc]].to(
                    self.device) @ self.embeds[str(ntype)].to(self.device)
            else:
                loc = node_tids == ntype
                embeds[loc] = self.node_embeds(tsd_ids[loc]).to(self.device)

        return embeds


@utils.benchmark('time', 600)
@utils.parametrize('data', ['am', 'ogbn-mag'])
def track_time(data):
    dataset = utils.process_data(data)
    device = utils.get_bench_device()

    if data == 'am':
        n_bases = 40
        l2norm = 5e-4
    elif data == 'ogbn-mag':
        n_bases = 2
        l2norm = 0
    else:
        raise ValueError()

    num_runs = 3
    fanouts = [25, 15]
    n_layers = 2
    batch_size = 64 if data == 'am' else 1024
    n_hidden = 64
    dropout = 0.5
    use_self_loop = True
    lr = 0.01
    low_mem = True
    num_workers = 4

    hg = dataset[0]
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = hg.nodes[category].data.pop('train_mask')
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    labels = hg.nodes[category].data.pop('labels').to(device)
    num_of_ntype = len(hg.ntypes)
    num_rels = len(hg.canonical_etypes)

    node_feats = []
    for ntype in hg.ntypes:
        if len(hg.nodes[ntype].data) == 0 or 'feat' not in hg.nodes[ntype].data:
            node_feats.append(None)
        else:
            feat = hg.nodes[ntype].data.pop('feat')
            node_feats.append(feat.share_memory_())

    # get target category id
    category_id = len(hg.ntypes)
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i
    g = dgl.to_homogeneous(hg)
    u, v, eid = g.all_edges(form='all')

    # global norm
    _, inverse_index, count = th.unique(
        v, return_inverse=True, return_counts=True)
    degrees = count[inverse_index]
    norm = th.ones(eid.shape[0]) / degrees
    norm = norm.unsqueeze(1)
    g.edata['norm'] = norm
    g.edata['etype'] = g.edata[dgl.ETYPE]
    g.ndata['type_id'] = g.ndata[dgl.NID]
    g.ndata['ntype'] = g.ndata[dgl.NTYPE]

    node_ids = th.arange(g.number_of_nodes())
    # find out the target node ids
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_nids = node_ids[loc]
    train_nids = target_nids[train_idx]

    g = g.formats('csc')
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    loader = dgl.dataloading.NodeDataLoader(
        g,
        target_nids[train_idx],
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers)

    # node features
    # None for one-hot feature, if not none, it should be the feature tensor.

    embed_layer = RelGraphEmbedLayer(device,
                                     g.number_of_nodes(),
                                     node_tids,
                                     num_of_ntype,
                                     node_feats,
                                     n_hidden,
                                     sparse_emb=True)

    # create model
    # all model params are in device.
    model = EntityClassify(device,
                           g.number_of_nodes(),
                           n_hidden,
                           num_classes,
                           num_rels,
                           num_bases=n_bases,
                           num_hidden_layers=n_layers - 2,
                           dropout=dropout,
                           use_self_loop=use_self_loop,
                           low_mem=low_mem,
                           layer_norm=False)

    embed_layer = embed_layer.to(device)
    model = model.to(device)

    all_params = itertools.chain(
        model.parameters(), embed_layer.embeds.parameters())
    optimizer = th.optim.Adam(all_params, lr=lr, weight_decay=l2norm)
    emb_optimizer = th.optim.SparseAdam(
        list(embed_layer.node_embeds.parameters()), lr=lr)

    timer = utils.ModelSpeedTimer()

    for run in range(num_runs):
        # dry run
        for i, sample_data in enumerate(loader):
            input_nodes, output_nodes, blocks = sample_data
            feats = embed_layer(input_nodes,
                                blocks[0].srcdata['ntype'],
                                blocks[0].srcdata['type_id'],
                                node_feats)
            logits = model(blocks, feats)
            seed_idx = blocks[-1].dstdata['type_id']
            loss = F.cross_entropy(logits, labels[seed_idx])
            optimizer.zero_grad()
            emb_optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            emb_optimizer.step()

            if i >= 4:
                break

        print("start training...")
        model.train()
        embed_layer.train()

        for i, sample_data in enumerate(loader):
            with timer as t:
                input_nodes, output_nodes, blocks = sample_data
                feats = embed_layer(input_nodes,
                                    blocks[0].srcdata['ntype'],
                                    blocks[0].srcdata['type_id'],
                                    node_feats)
                logits = model(blocks, feats)
                seed_idx = blocks[-1].dstdata['type_id']
                loss = F.cross_entropy(logits, labels[seed_idx])
                optimizer.zero_grad()
                emb_optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                emb_optimizer.step()

            if i >= 9:  # time 10 loops
                break

    return timer.average_epoch_time
