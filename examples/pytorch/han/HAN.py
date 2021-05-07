import time
import numpy as np
import argparse
import random
import torch.nn.functional as F
import torch.sparse
import dgl
from common_scripts import *

import torch.nn as nn
import torch.nn.functional as F
import scipy
from dgl.nn.pytorch import edge_softmax
import dgl.function as fn

import os, pickle, sys
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=64):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)

class HAN_model(nn.Module):
    def __init__(self,
                 num_layers,
                 target,
                 metapaths_list,
                 feats_dim,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 features_extractor,
                 dropout_rate=0.5):
        super(HAN_model, self).__init__()
        self.num_layers = num_layers
        self.target = target
        self.hidden_dim = hidden_dim
        self.metapaths_list = metapaths_list
        self.features_extractor = features_extractor;
        # ntype-specific transformation
        self.fc = nn.Linear(feats_dim, hidden_dim, bias=True);
        # feature dropout after trainsformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        
        self.semantic_attention = SemanticAttention(hidden_dim, hidden_dim)

        # hidden layers
        self.layers = nn.ModuleDict({})
        # hidden layers
        for metapath in self.metapaths_list:
            self.layers[metapath] = nn.ModuleList([]);
            for i in range(num_layers):
                self.layers[metapath].append(dgl.nn.pytorch.conv.GATConv(in_feats=(hidden_dim, hidden_dim), out_feats=hidden_dim//num_heads, num_heads=num_heads, feat_drop=0.0, attn_drop=dropout_rate, negative_slope=0.2, residual=False, activation=None));
        # output projection layer
        self.out_layer = MLP(0, hidden_dim, hidden_dim, out_dim)
    def forward(self, blocks):
        features = self.features_extractor.extract_features(blocks[0], which='src');
        h_ = self.feat_drop(self.fc(features[self.target]));
        h = h_;
        
        # hidden layers
        features_dict = {}
        for metapath in self.metapaths_list:
            h = h_;
            for l in range(self.num_layers):
                g = dgl.bipartite(blocks[l].adjacency_matrix(scipy_fmt='coo', etype=metapath, transpose=True));
                h = self.layers[metapath][l](g, (h, h[:g.number_of_dst_nodes()]))
                h = self.feat_drop(F.elu(h));
                h = h.view(h.size(0), -1)
            features_dict[metapath] = h;
            
        h = torch.stack([features_dict[x] for x in sorted(list(features_dict.keys()))], dim=1);
        h = self.semantic_attention(h) 
            
        # output projection layer
        logits = self.out_layer(h);

        # return only the target nodes' logits and embeddings
        return logits, features;
    
    
    
class HAN():
    def __init__(self, 
                 G=None, 
                 features_dict=None, 
                 ntypes_list=None, 
                 selected_etypes_list=None, 
                 etypes_list=None, 
                 etypes_complements=None, 
                 l1_metapaths_list=None, 
                 l1_metapaths_sequence_list=None, 
                 l2_metapaths_list=None, 
                 l2_metapaths_sequence_list=None, 
                 l2_metapaths_nodetypes_list=None, 
                 num_labels=None, 
                 labels=None, 
                 num_layers=1, 
                 hidden_dim=64, 
                 num_heads=1, 
                 attn_vec_dim=128, 
                 num_epochs=100, 
                 patience=5, 
                 batch_size=1024, 
                 samples=10, 
                 dropout_rate=0.5, 
                 learning_rate=0.001, 
                 weight_decay=0.0001, 
                 device=torch.device('cpu'), 
                 seed=1029384756, 
                 verbose=False,
                 save_postfix='RGCN',
                 target='target',
                 weighted_loss = True):
    
        self.relation_fanout_dict = {etype:0 for etype in G.etypes};
        for etype in etypes_list:
            self.relation_fanout_dict[etype] = samples;
        self.num_layers = num_layers;
        self.G = G;
        self.labels = torch.LongTensor(labels).to(device);
        
        if weighted_loss:
            self.class_weight = torch.FloatTensor([np.count_nonzero(labels == _)+0.0 for _ in range(num_labels)]).to(device)
            self.class_weight = torch.max(self.class_weight)/self.class_weight;
        else:
            self.class_weight = torch.FloatTensor([1.0 for _ in range(num_labels)]).to(device)
            
        self.target = target;
        self.batch_size = batch_size;
        self.seed = seed;
        self.verbose = verbose;
        self.save_postfix = save_postfix;
        self.num_epochs = num_epochs;
        self.patience = patience;
        self.device = device;
        
        self.metapath_fanout_dict = {tuple(l2_metapath_sequence):samples for l2_metapath_sequence in l2_metapaths_sequence_list};
        self.metapath2edge = {tuple(l2_metapaths_sequence_list[i]):l2_metapaths_list[i] for i in range(len(l2_metapaths_sequence_list))};
        features_dict = {ntype:torch.FloatTensor(features_dict[ntype]).to(device) for ntype in features_dict}
        in_dims = {ntype:features_dict[ntype].shape[1] for ntype in features_dict}
        block_features_extractor = features_extractor_class(device, features_dict);
        
        
        self.net = HAN_model(num_layers, target, l2_metapaths_list, in_dims[target], hidden_dim, num_labels, num_heads, block_features_extractor, dropout_rate);
        self.net.to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    def fit(self, train_indices, valid_indices):
        self.net.train()
        dur1 = []
        dur2 = []
        dur3 = []
        train_sampler = HANSampler(self.G, self.metapath_fanout_dict, self.metapath2edge, self.target, self.num_layers);
        loader = DataLoader(dataset=train_indices.tolist(), batch_size=self.batch_size, 
                                collate_fn=train_sampler.sample_blocks, shuffle=True, num_workers=0, worker_init_fn=np.random.seed(self.seed));
        
        # validation sampler
        val_blocks, val_labels = get_block_list_han(self.G, valid_indices, self.batch_size, self.target, self.metapath_fanout_dict, self.metapath2edge, self.num_layers, self.labels);
        
        early_stopping = EarlyStopping(patience=self.patience, verbose=self.verbose, save_path='checkpoint/checkpoint_{}.pt'.format(self.save_postfix))
        for epoch in range(self.num_epochs):
            for i, (seeds, blocks) in enumerate(loader):
                t0 = time.time()
                # training forward
                self.net.train()
                logits, embeddings = self.net(blocks)
                logp = F.log_softmax(logits, 1)
                train_loss = F.nll_loss(logp, self.labels[seeds], weight=self.class_weight)
                t1 = time.time()
                dur1.append(t1 - t0)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                preds = logits.argmax(1).cpu();
                train_macro_f1 = f1_score(self.labels.cpu()[seeds], preds, average='macro')
                train_micro_f1 = f1_score(self.labels.cpu()[seeds], preds, average='micro')
                t2 = time.time()
                dur2.append(t2 - t1)
                if self.verbose:
                    print("Epoch {:05d} | Batch {:03d} | Train macro F1: {:.4f} | Train micro F1: {:.4f} | Train Loss: {:.4f}".
                      format(epoch, i, train_macro_f1, train_micro_f1, train_loss.item()))
            # validation
            t2 = time.time()
            val_loss, val_macro_f1, val_micro_f1 = evaluate(self.net, val_blocks, val_labels, self.class_weight, self.device)
            t3 = time.time()
            early_stopping(val_loss, self.net)
            if early_stopping.early_stop:
                if self.verbose:
                    print('Early stopping!')
                break

            dur3.append(t3 - t2)
            if self.verbose:
                print("Epoch {:05d} | Validation macro F1: {:.4f} | Validation micro F1: {:.4f} | Validation Loss: {:.4f} | Forward propagation time: {:.4f} | Back propagation time Loss: {:.4f} | Validation time: {:.4f}".format(epoch, val_macro_f1, val_micro_f1, val_loss, np.mean(dur1), np.mean(dur2), np.mean(dur3)))
    
    def predict(self, test_indices):
        # sampler
        test_blocks, test_labels = get_block_list_han(self.G, test_indices, self.batch_size, self.target, self.metapath_fanout_dict, self.metapath2edge, self.num_layers, self.labels);
        self.net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(self.save_postfix)))
        test_loss, test_macro_f1, test_micro_f1 = evaluate(self.net, test_blocks, test_labels, self.class_weight, self.device);
        if self.verbose:
            print("Test macro F1: {:.4f} | Test micro F1: {:.4f} | Test Loss: {:.4f}".format(test_macro_f1, test_micro_f1, test_loss))
        return test_loss, test_macro_f1, test_micro_f1;
    
    def predictions(self, test_indices):
        # sampler
        test_blocks, test_labels = get_block_list_han(self.G, test_indices, self.batch_size, self.target, self.metapath_fanout_dict, self.metapath2edge, self.num_layers, self.labels);
        self.net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(self.save_postfix)))
        logits, preds = get_preds(self.net, test_blocks, test_labels, self.device);
        return logits, preds;
    
    
class HeteroNeighborSampler:
    def __init__(self, g, target, fanout_list):
        self.g = g
        self.target = target;
        self.fanout_list = fanout_list;
    def sample_blocks(self, seeds):
        blocks = []
        seeds = {self.target : seeds}
        cur = seeds
        for fanout in self.fanout_list:
            if fanout is None:
                frontier = dgl.in_subgraph(self.g, cur)
            else:
                frontier = dgl.sampling.sample_neighbors(self.g, cur, fanout)
            block = dgl.to_block(frontier, cur, include_dst_in_src=True)
            cur = {}
            for ntype in block.srctypes:
                cur[ntype] = block.srcnodes[ntype].data[dgl.NID]
            blocks.insert(0, block)
        return seeds, blocks
    
class MetapathSampler:
    def __init__(self, g, metapath2samples, metapath_graph, target):
        self.g = g
        self.target = target;
        self.metapath2samples = metapath2samples;
        self.metapath_graph = metapath_graph;
        
    def sample_blocks(self, seeds):
        seeds = {self.target : seeds};
        frontier = dgl.sampling.sample_neighbors(self.metapath_graph, seeds, self.metapath2samples)
        block = dgl.to_block(frontier, seeds, include_dst_in_src=True)
        
        return seeds[self.target], [block];
    
    
class L2NeighborSampler:
    def __init__(self, g, metapath2samples, metapath2edge, target, num_layers):
        self.g = g
        self.target = target;
        self.metapath2samples = metapath2samples;
        self.metapath2edge = metapath2edge;
        self.num_layers = num_layers;
        self.samplers = [{} for i in range(num_layers)];
        for i in range(num_layers):
            for metapath in metapath2samples:
                num_neighbors = metapath2samples[metapath];
                self.samplers[i][metapath] = dgl.sampling.RandomWalkNeighborSampler(self.g, random_walk_length=1, random_walk_restart_prob=0.0, num_random_walks=10*num_neighbors, num_neighbors=num_neighbors, metapath=metapath, weight_column='weights');

    def sample_blocks(self, seeds):
        blocks = []
        seeds = torch.LongTensor(seeds);
        cur = seeds
        
        for i in range(self.num_layers):
            rel_dict = {};
            for metapath in self.metapath2samples:
                sampled_g = self.samplers[i][metapath](cur);
                rel_dict[(self.target, self.metapath2edge[metapath], self.target)] = sampled_g.all_edges();
            frontier = dgl.heterograph(rel_dict);
            block = dgl.to_block(frontier, cur, include_dst_in_src=True)
            cur = block.srcnodes[self.target].data[dgl.NID]
            blocks.insert(0, block)
        return cur, blocks
    
class HANSampler:
    def __init__(self, g, metapath2samples, metapath2edge, target, num_layers):
        self.g = g
        self.target = target;
        self.metapath2samples = metapath2samples;
        self.metapath2edge = metapath2edge;
        self.num_layers = num_layers;
        self.samplers = [{} for i in range(num_layers)];
        for i in range(num_layers):
            for metapath in metapath2samples:
                num_neighbors = metapath2samples[metapath];
                self.samplers[i][metapath] = dgl.sampling.RandomWalkNeighborSampler(self.g, random_walk_length=1, random_walk_restart_prob=0.0, 
                                                                                                num_random_walks=10*num_neighbors, num_neighbors=num_neighbors, metapath=metapath, weight_column='weights');

    def sample_blocks(self, seeds):
        blocks = []
        seeds = torch.LongTensor(seeds);
        cur = seeds
        
        for i in range(self.num_layers):
            rel_dict = {};
            for metapath in self.metapath2samples:
                sampled_g = self.samplers[i][metapath](cur);
                rel_dict[(self.target, self.metapath2edge[metapath], self.target)] = sampled_g.all_edges();
            frontier = dgl.heterograph(rel_dict);
            block = dgl.to_block(frontier, cur, include_dst_in_src=True)
            cur = block.srcnodes[self.target].data[dgl.NID]
            blocks.insert(0, block)
        return seeds, blocks
    
def get_block_list_han(G, w_indices, batch_size, target, metapath_fanout_dict, metapath2edge, num_layers, labels):
    w_sampler = HANSampler(G, metapath_fanout_dict, metapath2edge, target, num_layers);
    w_loader = DataLoader(dataset=w_indices.tolist(), batch_size=batch_size, 
                            collate_fn=w_sampler.sample_blocks, shuffle=False, num_workers=0);
    block_list = [];
    label_list = [];
    for i, (seeds, blocks) in enumerate(w_loader):
        label_list.append(labels[seeds]);
        block_list.append(blocks);
    return block_list, label_list;

def metapath_reachable_graph(g, metapath, relation):
    """Return a graph where the successors of any node ``u`` are nodes reachable from ``u`` by
    the given metapath.
    If the beginning node type ``s`` and ending node type ``t`` are the same, it will return
    a homogeneous graph with node type ``s = t``.  Otherwise, a unidirectional bipartite graph
    with source node type ``s`` and destination node type ``t`` is returned.
    In both cases, two nodes ``u`` and ``v`` will be connected with an edge ``(u, v)`` if
    there exists one path matching the metapath from ``u`` to ``v``.
    The result graph keeps the node set of type ``s`` and ``t`` in the original graph even if
    they might have no neighbor.
    The features of the source/destination node type in the original graph would be copied to
    the new graph.
    Parameters
    ----------
    g : DGLHeteroGraph
        The input graph
    metapath : list[str or tuple of str]
        Metapath in the form of a list of edge types
    Returns
    -------
    DGLHeteroGraph
        A homogeneous or bipartite graph.
    """
    adj = 1
    index_dtype = g._graph.dtype
    for etype in metapath:
        adj = adj * g.adj(etype=etype, scipy_fmt='csr', transpose=True)

    adj = (adj != 0).tocsr()
    srctype = g.to_canonical_etype(metapath[0])[0]
    dsttype = g.to_canonical_etype(metapath[-1])[2]
    
    if srctype == dsttype:
        assert adj.shape[0] == adj.shape[1]
        new_g = dgl.graph(adj, ntype=srctype, index_dtype=index_dtype)
    else:
        new_g = dgl.bipartite(adj, utype=srctype, vtype=dsttype, etype=relation, index_dtype=index_dtype)

    return new_g

def get_metapath_heterograph(G, metapaths_list, relations_list):
    metapath2relation = [];
    for i in range(len(metapaths_list)):
        metapath = metapaths_list[i];
        adj = 1
        index_dtype = G._graph.dtype
        for etype in metapath:
            adj = adj * G.adj(etype=etype, scipy_fmt='csr', transpose=True)

        adj = (adj != 0).tocsr()
        srctype = G.to_canonical_etype(metapath[0])[0]
        dsttype = G.to_canonical_etype(metapath[-1])[2]
        new_g_1 = dgl.bipartite(adj, utype=srctype, vtype=dsttype, etype=relations_list[i], index_dtype=index_dtype)
        new_g_2 = dgl.bipartite(adj.transpose(), utype=dsttype, vtype=srctype, etype=dsttype+'__to__'+srctype, index_dtype=index_dtype)
        metapath2relation.append(new_g_1);
        metapath2relation.append(new_g_2);
    return dgl.hetero_from_relations(metapath2relation);


