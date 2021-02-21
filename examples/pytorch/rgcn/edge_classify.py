"""
Relational GCN for edge classification

Code follows the dgl tutorial:
https://docs.dgl.ai/guide/minibatch-edge.html#guide-minibatch-edge-classification-sampler

Data is generated by fake_data_generator.py

Graph: 2 types of nodes (seller and product). A seller-product pair is an offer listing, ie. edge. Task is to predict
whether an offer listing (edge) is defective.

Model: use RGCN propagation to get node embeddings, and then concatenate node embeddings with edge embeddings to make a
binary classification. The edge embeddings are generated from a neural network. All parameters are optimized as a whole
by the cross entropy objective function.

This code can handle:
1. Features on different node type have different dimensions.
2. Mini-batch training
3. RGCN homogeneous implementation.
4. Model performance is evaluated by accuracy and roc auc.

"""
import argparse
import numpy as np
import pandas as pd
import time
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import RelGraphConv
from sklearn.metrics import roc_auc_score

from model import RelGraphEmbedLayer
from entity_classify_mp import EntityClassify
from fake_data_generator import load_generated_data

class RGCN(EntityClassify):
    """ Edge classification class for RGCN
    Parameters
    ----------
    device : int
        Device to run the layer.
    feat_dim : int
        input feature dim size.
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
                 feat_dim,
                 h_dim,
                 out_dim,
                 num_rels,
                 num_bases=None,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False,
                 low_mem=True,
                 layer_norm=False):
        super().__init__(device,
                         None,
                         h_dim,
                         out_dim,
                         num_rels,
                         num_bases,
                         num_hidden_layers,
                         dropout,
                         use_self_loop,
                         low_mem,
                         layer_norm)

        self.feat_dim = feat_dim

        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConv(
            self.feat_dim, self.h_dim, self.num_rels, "basis",
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            low_mem=self.low_mem, dropout=self.dropout, layer_norm = layer_norm))
        # h2h
        for idx in range(self.num_hidden_layers):
            self.layers.append(RelGraphConv(
                self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                low_mem=self.low_mem, dropout=self.dropout, layer_norm = layer_norm))
        # h2o
        self.layers.append(RelGraphConv(
            self.h_dim, self.out_dim, self.num_rels, "basis",
            self.num_bases, activation=None,
            self_loop=self.use_self_loop,
            low_mem=self.low_mem, layer_norm = layer_norm))


class NeighborSampler:
    """Neighbor sampler
    Parameters
    ----------
    fanouts : list of int
        Fanout of each hop starting from the seed nodes. If a fanout is None,
        sample full neighbors.
    """
    def __init__(self, fanouts):
        self.fanouts = fanouts

    """Do neighbor sample
    Parameters
    ----------
    seeds :
        Seed nodes
    Returns
    -------
    tensor
        Seed nodes, also known as target nodes
    blocks
        Sampled subgraphs
    """

    def sample_blocks(self, g, seeds, *args, **kwargs):
        blocks = []
        cur = seeds
        for fanout in self.fanouts:
            if fanout is None or fanout == -1:
                frontier = dgl.in_subgraph(g, cur)
            else:
                frontier = dgl.sampling.sample_neighbors(g, cur, fanout)
            etypes = g.edata[dgl.ETYPE][frontier.edata[dgl.EID]]
            block = dgl.to_block(frontier, cur)
            block.srcdata[dgl.NTYPE] = g.ndata[dgl.NTYPE][block.srcdata[dgl.NID]]
            block.srcdata['type_id'] = g.ndata[dgl.NID][block.srcdata[dgl.NID]]
            block.edata['etype'] = etypes
            cur = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        return blocks


# edge score prediction
class ScorePredictor(nn.Module):
    def __init__(self, num_classes, in_features, edge_features=None, dropout=0.2):
        super().__init__()
        self.edge_features = edge_features

        if self.edge_features is not None:
            edge_feature_dim = edge_features.shape[1]
            h_dim = int(edge_feature_dim / 2)
            print("edge_feature_dim: ", edge_feature_dim)
            print("h_dim: ", h_dim)

            # add edge feature (edge features are converted to embeddings by a MLP
            self.linear_add = th.nn.Sequential(
                th.nn.Linear(edge_feature_dim, h_dim),
                th.nn.ReLU(),
                th.nn.Dropout(dropout)
            )

            self.linear = nn.Linear(2 * in_features + h_dim, num_classes)
        else:
            self.linear = nn.Linear(2 * in_features, num_classes)

    def apply_edges(self, edges):
        if self.edge_features is not None:
            offer_id = edges.data['type_id'].tolist()
            offer_features = self.edge_features[offer_id]
            x_add = self.linear_add(offer_features.float())
            x = th.cat([edges.src['x'], edges.dst['x'], x_add], dim=1)
        else:
            x = th.cat([edges.src['x'], edges.dst['x']], dim=1)

        return {'score': self.linear(x)}

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            edge_subgraph.apply_edges(self.apply_edges)
            return edge_subgraph.edata['score']


class EdgeClassify(nn.Module):
    def __init__(self,
                 device,
                 feat_dim,
                 h_dim,
                 out_dim,
                 num_rels,
                 edge_feats,
                 num_bases=None,
                 num_hidden_layers=1,
                 dropout=0.0,
                 use_self_loop=False,
                 low_mem=True,
                 layer_norm=False):
        super().__init__()

        self.rgcn = RGCN(
            device,
            feat_dim,
            h_dim,
            out_dim,
            num_rels,
            num_bases,
            num_hidden_layers,
            dropout,
            use_self_loop,
            low_mem,
            layer_norm)

        num_classes = 1
        self.predictor = ScorePredictor(num_classes, out_dim, edge_feats, dropout)

    def forward(self, edge_subgraph, blocks, x):
        x = self.rgcn(blocks, x)
        return self.predictor(edge_subgraph, x)


def evaluate(g, model, embed_layer, eval_loader, node_feats):
    model.eval()
    embed_layer.eval()
    eval_logits = []
    eval_labels = []
    eval_scores = []
    eval_loss_list = []

    with th.no_grad():
        for sample_data in eval_loader:
            input_nodes, edge_subgraph, blocks = sample_data
            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata[dgl.NTYPE],
                                blocks[0].srcdata['type_id'],
                                node_feats)
            edge_subgraph.edata['type_id'] = g.edata[dgl.EID][
                edge_subgraph.edata[dgl.EID]]  # add type_id to subgraph's edges.
            logits = model(edge_subgraph, blocks, feats)
            labels = edge_subgraph.edata["label"]

            eval_logits.append(logits.cpu().detach())
            eval_labels.append(labels.cpu().detach())

            eval_scores = eval_scores + list(th.sigmoid(logits).data.cpu().flatten().numpy())
            eval_loss_list.append(F.binary_cross_entropy_with_logits(logits.squeeze(), labels.double()).item())

    eval_logits = th.cat(eval_logits)
    eval_labels = th.cat(eval_labels)

    auc = roc_auc_score(eval_labels.numpy(), eval_scores)

    return eval_logits, eval_labels, auc, eval_loss_list


def accuracy(output, labels, threshold=0.5):
    """calculate prediction accuracy"""
    preds = torch.where(th.sigmoid(output) > threshold, torch.tensor(1, device=output.device),
                        torch.tensor(0, device=output.device)).type_as(labels)
    preds = preds.squeeze()
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train(dataset, args):
    g, e_ids_train, e_ids_val, e_ids_test, node_feats, edge_feats, num_rels, num_of_ntype = dataset

    # parameters
    dev_id = -1
    embed_size = args.n_hidden
    in_feats = embed_size
    n_hidden = embed_size
    n_output = embed_size
    num_bases = num_rels

    # total layer has additional input, output layer: 2 + num_hidden_layers
    num_hidden_layers = args.n_layers
    dropout = args.dropout
    use_self_loop = False
    fanouts = [15, 10, 5]

    assert (len(fanouts) == 2 + num_hidden_layers)

    node_tids = g.ndata[dgl.NTYPE]
    num_nodes = g.number_of_nodes()

    embed_layer = RelGraphEmbedLayer(dev_id,
                                     num_nodes,
                                     node_tids,
                                     num_of_ntype,
                                     node_feats,
                                     embed_size,
                                     dgl_sparse=args.dgl_sparse
                                     )

    edge_classification_model = EdgeClassify(
                                            device=-1,
                                            feat_dim=in_feats,
                                            h_dim=n_hidden,
                                            out_dim=n_output,
                                            num_rels=num_rels,
                                            edge_feats=edge_feats,
                                            num_bases=num_bases,
                                            num_hidden_layers=num_hidden_layers,
                                            dropout=dropout,
                                            use_self_loop=use_self_loop,
                                            low_mem=True,
                                            layer_norm=True)

    all_params = list(edge_classification_model.parameters()) + list(embed_layer.embeds.parameters())
    opt = torch.optim.Adam(all_params, lr=0.001)

    sampler = NeighborSampler(fanouts)
    dataloader = dgl.dataloading.EdgeDataLoader(g,
                                                e_ids_train,
                                                sampler,
                                                batch_size=args.graph_batch_size,
                                                shuffle=True,
                                                drop_last=False,
                                                num_workers=4)

    val_sampler = NeighborSampler(fanouts)
    val_loader = dgl.dataloading.EdgeDataLoader(g,
                                                e_ids_val,
                                                val_sampler,
                                                batch_size=args.eval_batch_size,
                                                shuffle=True,
                                                num_workers=4)

    test_sampler = NeighborSampler(fanouts)
    test_loader = dgl.dataloading.EdgeDataLoader(g,
                                                 e_ids_test,
                                                 test_sampler,
                                                 batch_size=args.eval_batch_size,
                                                 shuffle=True,
                                                 num_workers=4)

    # training
    epochs = args.n_epochs
    timeStart = time.time()
    for epoch in range(epochs):
        print(epoch)
        train_loss_list = []
        train_scores = []
        train_labels = []

        edge_classification_model.train()
        embed_layer.train()

        avg_loss = 0
        for i, (input_nodes, edge_subgraph, blocks) in enumerate(dataloader):
            input_features = embed_layer(blocks[0].srcdata[dgl.NID],
                                         blocks[0].srcdata[dgl.NTYPE],
                                         blocks[0].srcdata['type_id'],
                                         node_feats)
            edge_subgraph.edata['type_id'] = g.edata[dgl.EID][
                edge_subgraph.edata[dgl.EID]]  # add type_id to subgraph's edges.
            edge_labels = edge_subgraph.edata["label"]
            edge_predictions = edge_classification_model(edge_subgraph, blocks, input_features).squeeze()

            loss = F.binary_cross_entropy_with_logits(edge_predictions, edge_labels.double())

            train_loss_list.append(loss.item())
            train_scores = train_scores + list(th.sigmoid(edge_predictions).data.cpu().flatten().numpy())
            train_labels.append(edge_labels.cpu().detach())

            opt.zero_grad()
            loss.backward()
            opt.step()

            avg_loss += loss.item()
            if i % 1 == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item(),
                }
                print(str(post_fix), time.time() - timeStart)

        # Validation
        val_logits, val_labels, val_roc_auc, val_loss_list = evaluate(g, edge_classification_model, embed_layer,
                                                                      val_loader, node_feats)
        val_accuracy = accuracy(val_logits, val_labels)
        print('Train Epoch: {:01d}, lr: {}, train loss: {:.4f}, val loss: {:.4f}, val accuracy: {:.4f}, val roc auc: {:.4f}'.format(
                epoch, opt.param_groups[0]['lr'], np.mean(train_loss_list), np.mean(val_loss_list),
                val_accuracy, val_roc_auc))

    # Test
    test_logits, test_labels, test_roc_auc, test_loss_list = evaluate(g, edge_classification_model, embed_layer,
                                                                      test_loader, node_feats)
    test_accuracy = accuracy(test_logits, test_labels)
    print("test loss: {:.4f}, test_accuracy: {:.4f}, test_roc_auc: {:.4f} ".format(np.mean(test_loss_list), test_accuracy, test_roc_auc))


def main(args):
    # load graph data
    hg, node_feats, edge_feats, label_edge = load_generated_data()

    num_of_ntype = len(hg.ntypes)
    num_rels = len(hg.etypes)

    g = dgl.to_homogeneous(hg, edata=["label"])

    # get the edge type id of the edge that will be predicted(i.e. label_edge is "seller-asin" edge).
    # For example, there are 4 edge types on the graph, indexed by [0, 1, 2, 3]. the label_edge edge type id is 3.
    etype_id = hg.get_etype_id(label_edge)
    print("etype_id", etype_id)
    e_ids = np.where(g.edata[dgl.ETYPE].numpy().flatten() == etype_id)[0]  # get the e_ids on the homogeneous graph.
    e_ids = e_ids.tolist()

    np.random.seed(4)
    e_ids = np.random.permutation(e_ids)  # shuffle e_ids
    sample_size = int(len(e_ids) / 5)

    e_ids_test = e_ids[:sample_size]
    e_ids_val = e_ids[sample_size: 2 * sample_size]
    e_ids_train = e_ids[2 * sample_size:]

    e_ids_test.sort()
    e_ids_val.sort()
    e_ids_train.sort()

    print(e_ids_test)
    print(e_ids_val)
    print(e_ids_train)

    if args.global_norm:
        u, v, eid = g.all_edges(form='all')
        _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = th.ones(eid.shape[0]) / degrees
        norm = norm.unsqueeze(1)
        g.edata['norm'] = norm

    g.ndata[dgl.NTYPE].share_memory_()
    g.edata[dgl.ETYPE].share_memory_()
    g.edata['norm'].share_memory_()

    dataset = (g, e_ids_train, e_ids_val, e_ids_test, node_feats, edge_feats, num_rels, num_of_ntype)
    train(dataset, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rgcn edge classification')
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=500,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=100,
            help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=10,
            help="number of minimum training epochs")
    parser.add_argument("--eval-batch-size", type=int, default=8,
            help="batch size when evaluating")
    parser.add_argument("--global-norm", type=float, default=1.0,
            help="use global norm")
    parser.add_argument("--graph-batch-size", type=int, default=10,
            help="number of edges to sample in each iteration")
    parser.add_argument("--evaluate-every", type=int, default=5,
            help="perform evaluation every n epochs")
    parser.add_argument("--dgl-sparse", default=False, action='store_true',
            help='Use sparse embedding for node embeddings.')

    args = parser.parse_args()
    print(args)
    main(args)




