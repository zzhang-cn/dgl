import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import tqdm

def normalized_laplacian_edata(g, weight=None):
    with g.local_scope():
        if weight is None:
            weight = 'W'
            g.edata[weight] = torch.ones(g.number_of_edges(), device=g.device)
        g_rev = dgl.reverse(g, copy_edata=True)
        g.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'v'))
        g_rev.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'u'))
        g.ndata['u'] = g_rev.ndata['u']
        g.apply_edges(lambda edges: {'w': edges.data[weight] / torch.sqrt(edges.src['u'] * edges.dst['v'])})
        return g.edata['w']


def normalized_edata(g, weight=None):
    with g.local_scope():
        if weight is None:
            weight = 'W'
            g.edata[weight] = torch.ones(g.number_of_edges(), device=g.device)
        g.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'v'))
        g.apply_edges(lambda edges: {'w': 1 / edges.dst['v']})
        return g.edata['w']


class GraphConv(nn.Module):
    def __init__(self, in_feats, n_classes):
        super().__init__()
        self.W = nn.Linear(in_feats, n_classes)

    def forward(self, g, x, w):
        with g.local_scope():
            g.srcdata['x'] = self.W(x)
            g.edata['w'] = w
            g.update_all(fn.u_mul_e('x', 'w', 'm'), fn.sum('m', 'y'))
            return g.dstdata['y']


class SAGEConv(nn.Module):
    def __init__(self, in_feats, n_classes):
        super().__init__()
        self.W = nn.Linear(in_feats * 2, n_classes)

    def forward(self, g, x, w):
        with g.local_scope():
            g.srcdata['x'] = x
            g.dstdata['x'] = x[:g.number_of_dst_nodes()]
            #g.edata['w'] = w
            #g.update_all(fn.u_mul_e('x', 'w', 'm'), fn.sum('m', 'y'))
            g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'y'))
            h = torch.cat([g.dstdata['x'], g.dstdata['y']], 1)
            return self.W(h)


class Model(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, conv=GraphConv, dropout=0):
        super().__init__()

        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        self.convs.append(conv(in_feats, n_hidden))
        for i in range(n_layers - 2):
            self.convs.append(conv(n_hidden, n_hidden))
        self.convs.append(conv(n_hidden, n_classes))

    def forward(self, blocks, x):
        if not isinstance(blocks, list):
            blocks = [blocks] * len(self.convs)
        for i, (conv, block) in enumerate(zip(self.convs, blocks)):
            x = self.dropout(x)
            x = conv(block, x, block.edata['w'])
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x

    def inference(self, g, x, w, batch_size, device, num_workers):
        with torch.no_grad():
            for l, layer in enumerate(self.convs):
                y = torch.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.convs) - 1 else self.n_classes)

                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                dataloader = dgl.dataloading.NodeDataLoader(
                    g,
                    torch.arange(g.number_of_nodes()),
                    sampler,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=num_workers)

                for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                    block = blocks[0]

                    block = block.int().to(device)
                    h = x[input_nodes].to(device)
                    w_block = w[block.edata[dgl.EID]].to(device)
                    h = layer(block, h, w_block)
                    if l != len(self.convs) - 1:
                        h = F.relu(h)

                    y[output_nodes] = h.cpu()

                x = y
            return y
