import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from .. import utils


def _init_input_modules(g, ntype, textset, hidden_dims):
    # We initialize the linear projections of each input feature ``x`` as
    # follows:
    # * If ``x`` is a scalar integral feature, we assume that ``x`` is a categorical
    #   feature, and assume the range of ``x`` is 0..max(x).
    # * If ``x`` is a float one-dimensional feature, we assume that ``x`` is a
    #   numeric vector.
    # * If ``x`` is a field of a textset, we process it as bag of words.
    module_dict = nn.ModuleDict()

    for column, data in g.nodes[ntype].data.items():
        if column == dgl.NID:
            continue
        if data.dtype == torch.float32:
            assert data.ndim == 2
            m = nn.Linear(data.shape[1], hidden_dims)
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
            module_dict[column] = m
        elif data.dtype == torch.int64:
            assert data.ndim == 1
            m = nn.Embedding(
                data.max() + 2, hidden_dims, padding_idx=-1)
            nn.init.xavier_uniform_(m.weight)
            module_dict[column] = m

    if textset is not None:
        for column, field in textset.fields.items():
            if field.vocab.vectors:
                module_dict[column] = BagOfWordsPretrained(field, hidden_dims)
            else:
                module_dict[column] = BagOfWords(field, hidden_dims)

    return module_dict


class BagOfWordsPretrained(nn.Module):
    def __init__(self, field, hidden_dims):
        super().__init__()

        input_dims = field.vocab.vectors.shape[1]
        self.emb = nn.Embedding(
            len(field.vocab.itos), input_dims,
            padding_idx=field.vocab.stoi[field.pad_token])
        self.emb.weight[:] = field.vocab.vectors
        self.proj = nn.Linear(input_dims, hidden_dims)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x, length):
        """
        x: (batch_size, max_length) LongTensor
        length: (batch_size,) LongTensor
        """
        x = self.emb(x).sum(1) / length.unsqueeze(1).float()
        return self.proj(x)


class BagOfWords(nn.Module):
    def __init__(self, field, hidden_dims):
        super().__init__()

        self.emb = nn.Embedding(
            len(field.vocab.itos), hidden_dims,
            padding_idx=field.vocab.stoi[field.pad_token])
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, x, length):
        return self.emb(x).sum(1) / length.unsqueeze(1).float()


class WeightedSAGEConv(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, act=F.relu):
        super().__init__()

        self.act = act
        self.Q = nn.Linear(input_dims, hidden_dims)
        self.W = nn.Linear(input_dims + hidden_dims, output_dims)
        self.reset_parameters()
        self.dropout = nn.Dropout(0.5)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, g, h, weights):
        """
        g : graph
        h : node features
        weights : scalar edge weights
        """
        h_src, h_dst = h
        with g.local_scope():
            g.srcdata['n'] = self.act(self.Q(self.dropout(h_src)))
            g.edata['w'] = weights.float()
            g.update_all(fn.u_mul_e('n', 'w', 'm'), fn.sum('m', 'n'))
            g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'ws'))
            n = g.dstdata['n']
            ws = g.dstdata['ws'].unsqueeze(1).clamp(min=1)
            z = self.act(self.W(self.dropout(torch.cat([n / ws, h_dst], 1))))
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(
                z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
            z = z / z_norm
            return z


class SAGENet(nn.Module):
    def __init__(self, hidden_dims, n_layers):
        """
        g : DGLHeteroGraph
            The user-item interaction graph.
            This is only for finding the range of categorical variables.
        item_textsets : torchtext.data.Dataset
            The textual features of each item node.
        """
        super().__init__()

        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(WeightedSAGEConv(
                hidden_dims, hidden_dims, hidden_dims))

    def forward(self, blocks, h):
        for layer, block in zip(self.convs, blocks):
            h_dst = h[:block.number_of_nodes('DST/' + block.ntypes[0])]
            h = layer(block, (h, h_dst), block.edata['weights'])
        return h


class LinearProjector(nn.Module):
    """
    Projects each input feature of the graph linearly and sums them up
    """

    def __init__(self, full_graph, ntype, textset, hidden_dims):
        super().__init__()

        self.ntype = ntype
        self.inputs = _init_input_modules(
            full_graph, ntype, textset, hidden_dims)

    def forward(self, ndata):
        projections = []
        for feature, data in ndata.items():
            if feature == dgl.NID or feature.endswith('__len'):
                # This is an additional feature indicating the length of the ``feature``
                # column; we shouldn't process this.
                continue

            module = self.inputs[feature]
            if isinstance(module, (BagOfWords, BagOfWordsPretrained)):
                # Textual feature; find the length and pass it to the textual module.
                length = ndata[feature + '__len']
                result = module(data, length)
            else:
                result = module(data)
            projections.append(result)

        return torch.stack(projections, 1).sum(1)


class ItemToItemScorer(nn.Module):
    def __init__(self, full_graph, ntype):
        super().__init__()

        n_nodes = full_graph.number_of_nodes(ntype)
        self.bias = nn.Parameter(torch.zeros(n_nodes))

    def _add_bias(self, edges):
        bias_src = self.bias[edges.src[dgl.NID]]
        bias_dst = self.bias[edges.dst[dgl.NID]]
        return {'s': edges.data['s'] + bias_src + bias_dst}

    def forward(self, item_item_graph, h):
        """
        item_item_graph : graph consists of edges connecting the pairs
        h : hidden state of every node
        """
        with item_item_graph.local_scope():
            item_item_graph.ndata['h'] = h
            item_item_graph.apply_edges(fn.u_dot_v('h', 'h', 's'))
            item_item_graph.apply_edges(self._add_bias)
            pair_score = item_item_graph.edata['s']
        return pair_score


class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, textsets, hidden_dims, n_layers):
        super().__init__()

        self.proj = LinearProjector(full_graph, ntype, textsets, hidden_dims)
        self.sage = SAGENet(hidden_dims, n_layers)
        self.scorer = ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)


def compact_and_copy(frontier, seeds):
    block = dgl.to_block(frontier, seeds)
    for col, data in frontier.edata.items():
        if col == dgl.EID:
            continue
        block.edata[col] = data[block.edata[dgl.EID]]
    return block


class ItemToItemBatchSampler(IterableDataset):
    def __init__(self, g, user_type, item_type, batch_size):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            heads = torch.randint(0, self.g.number_of_nodes(
                self.item_type), (self.batch_size,))
            tails = dgl.sampling.random_walk(
                self.g,
                heads,
                metapath=[self.item_to_user_etype, self.user_to_item_etype])[0][:, 2]
            neg_tails = torch.randint(0, self.g.number_of_nodes(
                self.item_type), (self.batch_size,))

            mask = (tails != -1)
            yield heads[mask], tails[mask], neg_tails[mask]


class NeighborSampler(object):
    def __init__(self, g, user_type, item_type, random_walk_length, random_walk_restart_prob,
                 num_random_walks, num_neighbors, num_layers):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.samplers = [
            dgl.sampling.PinSAGESampler(g, item_type, user_type, random_walk_length,
                                        random_walk_restart_prob, num_random_walks, num_neighbors)
            for _ in range(num_layers)]

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        blocks = []
        for sampler in self.samplers:
            frontier = sampler(seeds)
            if heads is not None:
                eids = frontier.edge_ids(torch.cat([heads, heads]), torch.cat(
                    [tails, neg_tails]), return_uv=True)[2]
                if len(eids) > 0:
                    old_frontier = frontier
                    frontier = dgl.remove_edges(old_frontier, eids)
                    # print(old_frontier)
                    # print(frontier)
                    # print(frontier.edata['weights'])
                    #frontier.edata['weights'] = old_frontier.edata['weights'][frontier.edata[dgl.EID]]
            block = compact_and_copy(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks

    def sample_from_item_pairs(self, heads, tails, neg_tails):
        # Create a graph with positive connections only and another graph with negative
        # connections only.
        pos_graph = dgl.graph(
            (heads, tails),
            num_nodes=self.g.number_of_nodes(self.item_type))
        neg_graph = dgl.graph(
            (heads, neg_tails),
            num_nodes=self.g.number_of_nodes(self.item_type))
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        seeds = pos_graph.ndata[dgl.NID]

        blocks = self.sample_blocks(seeds, heads, tails, neg_tails)
        return pos_graph, neg_graph, blocks


def assign_simple_node_features(ndata, g, ntype, assign_id=False):
    """
    Copies data to the given block from the corresponding nodes in the original graph.
    """
    for col in g.nodes[ntype].data.keys():
        if not assign_id and col == dgl.NID:
            continue
        induced_nodes = ndata[dgl.NID]
        ndata[col] = g.nodes[ntype].data[col][induced_nodes]


def assign_textual_node_features(ndata, textset, ntype):
    """
    Assigns numericalized tokens from a torchtext dataset to given block.

    The numericalized tokens would be stored in the block as node features
    with the same name as ``field_name``.

    The length would be stored as another node feature with name
    ``field_name + '__len'``.

    block : DGLHeteroGraph
        First element of the compacted blocks, with "dgl.NID" as the
        corresponding node ID in the original graph, hence the index to the
        text dataset.

        The numericalized tokens (and lengths if available) would be stored
        onto the blocks as new node features.
    textset : torchtext.data.Dataset
        A torchtext dataset whose number of examples is the same as that
        of nodes in the original graph.
    """
    node_ids = ndata[dgl.NID].numpy()

    for field_name, field in textset.fields.items():
        examples = [getattr(textset[i], field_name) for i in node_ids]

        tokens, lengths = field.process(examples)

        if not field.batch_first:
            tokens = tokens.t()

        ndata[field_name] = tokens
        ndata[field_name + '__len'] = lengths


def assign_features_to_blocks(blocks, g, textset, ntype):
    # For the first block (which is closest to the input), copy the features from
    # the original graph as well as the texts.
    assign_simple_node_features(blocks[0].srcdata, g, ntype)
    assign_textual_node_features(blocks[0].srcdata, textset, ntype)
    assign_simple_node_features(blocks[-1].dstdata, g, ntype)
    assign_textual_node_features(blocks[-1].dstdata, textset, ntype)


class PinSAGECollator(object):
    def __init__(self, sampler, g, ntype, textset):
        self.sampler = sampler
        self.ntype = ntype
        self.g = g
        self.textset = textset

    def collate_train(self, batches):
        heads, tails, neg_tails = batches[0]
        # Construct multilayer neighborhood via PinSAGE...
        pos_graph, neg_graph, blocks = self.sampler.sample_from_item_pairs(
            heads, tails, neg_tails)
        assign_features_to_blocks(blocks, self.g, self.textset, self.ntype)

        return pos_graph, neg_graph, blocks

    def collate_test(self, samples):
        batch = torch.LongTensor(samples)
        blocks = self.sampler.sample_blocks(batch)
        assign_features_to_blocks(blocks, self.g, self.textset, self.ntype)
        return blocks


@utils.benchmark('time', 600)
@utils.parametrize('data', ['nowplaying_rs'])
def track_time(data):
    dataset = utils.process_data(data)
    device = utils.get_bench_device()

    user_ntype = dataset.user_ntype
    item_ntype = dataset.item_ntype
    textset = dataset.textset

    num_runs = 3
    batch_size = 32
    random_walk_length = 2
    random_walk_restart_prob = 0.5
    num_random_walks = 10
    num_neighbors = 3
    num_layers = 2
    num_workers = 0
    hidden_dims = 16
    lr = 3e-5

    g = dataset[0]
    # Sampler
    batch_sampler = ItemToItemBatchSampler(
        g, user_ntype, item_ntype, batch_size)
    neighbor_sampler = NeighborSampler(
        g, user_ntype, item_ntype, random_walk_length,
        random_walk_restart_prob, num_random_walks, num_neighbors,
        num_layers)
    collator = PinSAGECollator(neighbor_sampler, g, item_ntype, textset)
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=num_workers)
    dataloader_test = DataLoader(
        torch.arange(g.number_of_nodes(item_ntype)),
        batch_size=batch_size,
        collate_fn=collator.collate_test,
        num_workers=num_workers)

    # Model
    model = PinSAGEModel(g, item_ntype, textset,
                         hidden_dims, num_layers).to(device)
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    timer = utils.ModelSpeedTimer()

    for run in range(num_runs):
        for batch_id, (pos_graph, neg_graph, blocks) in enumerate(dataloader):
            # Copy to GPU
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            loss = model(pos_graph, neg_graph, blocks).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            if batch_id >= 4:
                break

        print("start training...")

        # For each batch of head-tail-negative triplets...
        for batch_id, (pos_graph, neg_graph, blocks) in enumerate(dataloader):
            with timer as t:
                # Copy to GPU
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)
                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)

                loss = model(pos_graph, neg_graph, blocks).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

            if batch_id >= 9:  # time 10 loops
                break

    return timer.average_epoch_time
