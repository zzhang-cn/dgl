import itertools

import dgl
import dgl.nn.pytorch as dglnn
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .. import utils


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(
                in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis(
                    (in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(
                    th.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(
                    self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation

        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.

        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(th.split(weight, 1, dim=0))}
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(
                k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)
        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class RelGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph."""

    def __init__(self,
                 g,
                 device,
                 embed_size,
                 num_nodes,
                 node_feats,
                 embed_name='embed',
                 activation=None,
                 dropout=0.0):
        super(RelGraphEmbed, self).__init__()
        self.g = g
        self.device = device
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.node_feats = node_feats

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        self.node_embeds = nn.ModuleDict()
        for ntype in g.ntypes:
            if node_feats[ntype] is None:
                sparse_emb = th.nn.Embedding(
                    num_nodes[ntype], embed_size, sparse=True)
                nn.init.uniform_(sparse_emb.weight, -1.0, 1.0)
                self.node_embeds[ntype] = sparse_emb
            else:
                input_emb_size = node_feats[ntype].shape[1]
                embed = nn.Parameter(th.Tensor(input_emb_size, embed_size))
                nn.init.xavier_uniform_(embed)
                self.embeds[ntype] = embed

    def forward(self, block=None):
        """Forward computation

        Parameters
        ----------
        block : DGLHeteroGraph, optional
            If not specified, directly return the full graph with embeddings stored in
            :attr:`embed_name`. Otherwise, extract and store the embeddings to the block
            graph and return.

        Returns
        -------
        DGLHeteroGraph
            The block graph fed with embeddings.
        """
        embeds = {}
        for ntype in block.ntypes:
            if self.node_feats[ntype] is None:
                embeds[ntype] = self.node_embeds[ntype](
                    block.nodes(ntype)).to(self.device)
            else:
                embeds[ntype] = self.node_feats[ntype][block.nodes(
                    ntype)].to(self.device) @ self.embeds[ntype]
        return embeds


class EntityClassify(nn.Module):
    def __init__(self,
                 g,
                 h_dim, out_dim,
                 num_bases,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False):
        super(EntityClassify, self).__init__()
        self.g = g
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.h_dim, self.rel_names,
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            dropout=self.dropout, weight=False))
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvLayer(
                self.h_dim, self.h_dim, self.rel_names,
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))
        # h2o
        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.out_dim, self.rel_names,
            self.num_bases, activation=None,
            self_loop=self.use_self_loop))

    def forward(self, h, blocks):
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        return h


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
    fanout = 4
    n_layers = 2
    batch_size = 64 if data == 'am' else 1024
    n_hidden = 64
    dropout = 0.5
    use_self_loop = True
    lr = 0.01

    hg = dataset[0]
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = hg.nodes[category].data.pop('train_mask')
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    labels = hg.nodes[category].data.pop('labels')

    node_feats = {}
    num_nodes = {}

    for ntype in hg.ntypes:
        node_feats[ntype] = hg.nodes[ntype].data['feat'] if 'feat' in hg.nodes[ntype].data else None
        num_nodes[ntype] = hg.number_of_nodes(ntype)

    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [fanout] * n_layers)
    loader = dgl.dataloading.NodeDataLoader(
        hg, {category: train_idx}, sampler,
        batch_size=batch_size, shuffle=True, num_workers=4)

    embed_layer = RelGraphEmbed(
        hg, device, n_hidden, num_nodes, node_feats)
    model = EntityClassify(hg,
                           n_hidden,
                           num_classes,
                           num_bases=n_bases,
                           num_hidden_layers=n_layers - 2,
                           dropout=dropout,
                           use_self_loop=use_self_loop)
    embed_layer = embed_layer.to(device)
    model = model.to(device)

    all_params = itertools.chain(
        model.parameters(), embed_layer.embeds.parameters())
    optimizer = th.optim.Adam(all_params, lr=lr, weight_decay=l2norm)
    sparse_optimizer = th.optim.SparseAdam(
        list(embed_layer.node_embeds.parameters()), lr=lr)

    timer = utils.ModelSpeedTimer()

    for run in range(num_runs):
        # dry run
        for i, (input_nodes, seeds, blocks) in enumerate(loader):
            blocks = [blk.to(device) for blk in blocks]
            # we only predict the nodes with type "category"
            seeds = seeds[category]
            emb = embed_layer(blocks[0])
            lbl = labels[seeds].to(device)
            emb = {k: e.to(device) for k, e in emb.items()}
            logits = model(emb, blocks)[category]
            loss = F.cross_entropy(logits, lbl)
            loss.backward()
            optimizer.step()
            sparse_optimizer.step()

            if i >= 4:
                break

        print("start training...")
        model.train()
        embed_layer.train()
        optimizer.zero_grad()
        sparse_optimizer.zero_grad()

        for i, (input_nodes, seeds, blocks) in enumerate(loader):
            with timer as t:
                blocks = [blk.to(device) for blk in blocks]
                # we only predict the nodes with type "category"
                seeds = seeds[category]
                emb = embed_layer(blocks[0])
                lbl = labels[seeds].to(device)
                emb = {k: e.to(device) for k, e in emb.items()}
                logits = model(emb, blocks)[category]
                loss = F.cross_entropy(logits, lbl)
                loss.backward()
                optimizer.step()
                sparse_optimizer.step()

            if i >= 9:  # time 10 loops
                break

    return timer.average_epoch_time
