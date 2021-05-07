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

class MAGNN_metapath_specific(nn.Module):
    def __init__(self,
                 etypes,
                 etypes_mapping,
                 out_dim,
                 num_heads,
                 etypes_complements,
                 rnn_type='gru',
                 r_vec=None,
                 attn_drop=0.5,
                 alpha=0.01):
        super(MAGNN_metapath_specific, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.etypes_complements = etypes_complements
        self.rnn_type = rnn_type
        self.etypes = etypes
        self.etypes_mapping = etypes_mapping
        self.r_vec = r_vec

        # rnn-like metapath instance aggregator
        # consider multiple attention heads
            
        self.attn = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        # weight initialization
        nn.init.xavier_normal_(self.attn.data, gain=1.414)

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        # Dropout attention scores and save them
        g.edata['a_drop'] = self.attn_drop(attention)

    def message_passing(self, edges):
        ft = edges.data['eft'] * edges.data['a_drop']
        return {'ft': ft}

    def forward(self, g, features_list):
        g = g.local_var();
        # features: num_paths x path_len x out_dim
        if self.rnn_type == 'TransE0' or self.rnn_type == 'TransE1':
            r_vec = self.r_vec
            if self.rnn_type == 'TransE0':
                r_vec = torch.stack((r_vec, -r_vec), dim=1)
                r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1])  # etypes x out_dim
                        
            features_list = [F.normalize(edata, p=2, dim=1) for edata in features_list];
            for i in range(len(self.etypes) - 1):
                temp_etypes = [self.etypes_mapping[etype] for etype in self.etypes[i:]]
                features_list[i] = features_list[i] + r_vec[temp_etypes].sum(dim=0)
            hidden = torch.mean(torch.stack(features_list), dim=0)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'RotatE0' or self.rnn_type == 'RotatE1':
            r_vec = F.normalize(self.r_vec, p=2, dim=2)
            if self.rnn_type == 'RotatE0':
                r_vec = torch.stack((r_vec, r_vec), dim=1)
                r_vec[:, 1, :, 1] = -r_vec[:, 1, :, 1]
                r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1], 2)  # etypes x out_dim/2 x 2
            features_list = [x.reshape(x.shape[0], x.shape[1] // 2, 2) for x in features_list];
            final_r_vec = torch.zeros([len(self.etypes), self.out_dim // 2, 2], device=r_vec.device)
            final_r_vec[-1, :, 0] = 1
            for i in range(final_r_vec.shape[0] - 2, -1, -1):
                final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes_mapping[self.etypes[i]], :, 0] -\
                                       final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes_mapping[self.etypes[i]], :, 1]
                final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes_mapping[self.etypes[i]], :, 1] +\
                                       final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes_mapping[self.etypes[i]], :, 0]
            for i in range(len(self.etypes) - 1):
                temp1 = features_list[i][:,:,0].clone() * final_r_vec[i, :, 0] -\
                        features_list[i][:,:,1].clone() * final_r_vec[i, :, 1]
                temp2 = features_list[i][:,:,0].clone() * final_r_vec[i, :, 1] +\
                        features_list[i][:,:,1].clone() * final_r_vec[i, :, 0]
                features_list[i][:,:,0] = temp1
                features_list[i][:,:,1] = temp2
            features_list = [x.reshape(x.shape[0], -1) for x in features_list];
            hidden = torch.mean(torch.stack(features_list), dim=0)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)

        eft = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.out_dim)  # E x num_heads x out_dim
        
        a = (eft * self.attn).sum(dim=-1).unsqueeze(dim=-1)  # E x num_heads x 1
        a = self.leaky_relu(a)
        g.edata.update({'eft': eft, 'a': a})
        # compute softmax normalized attention values
        self.edge_softmax(g)
        # compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        g.update_all(self.message_passing, fn.sum('ft', 'ft'))
        ret = g.dstdata['ft']  # E x num_heads x out_dim
        
        return ret
class MAGNN_model(nn.Module):
    def __init__(self,
                 num_layers,
                 target,
                 metapaths_list,
                 metapaths_sequence_dict,
                 metapaths_nodetypes_list,
                 etypes_list,
                 etypes_complements,
                 feats_dim_dict,
                 hidden_dim,
                 attn_vec_dim,
                 out_dim,
                 num_heads,
                 features_extractor,
                 rnn_type,
                 dropout_rate=0.5):
        super(MAGNN_model, self).__init__()
        self.num_layers = num_layers
        self.target = target
        self.hidden_dim = hidden_dim
        self.metapaths_list = metapaths_list
        self.metapaths_sequence_dict = metapaths_sequence_dict
        self.metapaths_nodetypes_list = metapaths_nodetypes_list
        self.etypes_list = etypes_list
        self.etypes_complements = etypes_complements;
        self.features_extractor = features_extractor;
        self.etypes_mapping = {};
        i = 0;
        for _ in self.etypes_complements:
            self.etypes_mapping[self.etypes_complements[_]] = i;
            self.etypes_mapping[_] = i+len(self.etypes_complements);
            i += 1;
        # ntype-specific transformation
        self.fc_dict = nn.ModuleDict({ntype: nn.Linear(feats_dim_dict[ntype], hidden_dim, bias=True) for ntype in feats_dim_dict})
        # feature dropout after trainsformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_dict:
            nn.init.xavier_normal_(self.fc_dict[fc].weight, gain=1.414)
        # etype-specific parameters
        
        if rnn_type == 'TransE0':
            r_vec = nn.Parameter(torch.empty(size=(len(etypes_complements), hidden_dim)))
        elif rnn_type == 'TransE1':
            r_vec = nn.Parameter(torch.empty(size=(len(etypes_list), hidden_dim)))
        elif rnn_type == 'RotatE0':
            r_vec = nn.Parameter(torch.empty(size=(len(etypes_complements), hidden_dim // 2, 2)))
        elif rnn_type == 'RotatE1':
            r_vec = nn.Parameter(torch.empty(size=(len(etypes_list), hidden_dim // 2, 2)))
        if r_vec is not None:
            nn.init.xavier_normal_(r_vec.data, gain=1.414)
                
        self.layers = nn.ModuleDict({})
        # hidden layers
        for metapath in self.metapaths_list:
            self.layers[metapath] = nn.ModuleList([]);
            for i in range(num_layers):
                self.layers[metapath].append(MAGNN_metapath_specific(etypes=self.metapaths_sequence_dict[metapath], etypes_mapping=self.etypes_mapping, out_dim=hidden_dim, num_heads=num_heads, etypes_complements=etypes_complements, rnn_type=rnn_type, r_vec=r_vec));
        
        self.metapath_attention_fc1 = nn.ModuleList([]);
        self.metapath_attention_fc2 = nn.ModuleList([]);
        self.metapath_attention_fc3 = nn.ModuleList([]);
        for i in range(num_layers):
            # metapath-level attention
            # note that the acutal input dimension should consider the number of heads
            # as multiple head outputs are concatenated together
            self.metapath_attention_fc1.append(nn.Linear(hidden_dim * num_heads, attn_vec_dim, bias=True));
            self.metapath_attention_fc2.append(nn.Linear(attn_vec_dim, 1, bias=False))
            self.metapath_attention_fc3.append(nn.Linear(hidden_dim * num_heads, hidden_dim, bias=True))

            # weight initialization
            nn.init.xavier_normal_(self.metapath_attention_fc1[-1].weight, gain=1.414)
            nn.init.xavier_normal_(self.metapath_attention_fc2[-1].weight, gain=1.414)
            nn.init.xavier_normal_(self.metapath_attention_fc3[-1].weight, gain=1.414)


        # output projection layer
        self.out_layer = MLP(0, hidden_dim, hidden_dim, out_dim)
    def forward(self, blocks):
        for l in range(len(blocks)):
            g, traces_dict = blocks[l];
            metapath_outs = [];
            for metapath in traces_dict:
                features = self.features_extractor.features_tensor(traces_dict[metapath], self.metapaths_nodetypes_list[metapath]);
                h = [self.feat_drop(self.fc_dict[self.metapaths_nodetypes_list[metapath][i]](features[i])) for i in range(len(features))]
                homo_g = dgl.bipartite(g.adjacency_matrix(scipy_fmt='coo', etype=metapath, transpose=True));
                if l > 0:
                    src_data = g.all_edges(form='uv', order='eid', etype=metapath)[0];
                    h[0] = h_[src_data];
                h = self.layers[metapath][l](homo_g, h);
                h = self.feat_drop(F.elu(h));
                metapath_outs.append(h.view(h.size(0), -1));
            beta = []
            for metapath_out in metapath_outs:
                fc1 = torch.tanh(self.metapath_attention_fc1[l](metapath_out))
                fc1_mean = torch.mean(fc1, dim=0)
                fc2 = self.metapath_attention_fc2[l](fc1_mean)
                beta.append(fc2)
            beta = torch.cat(beta, dim=0)
            beta = F.softmax(beta, dim=0)
            beta = torch.unsqueeze(beta, dim=-1)
            beta = torch.unsqueeze(beta, dim=-1)
            metapath_outs = [torch.unsqueeze(metapath_out, dim=0) for metapath_out in metapath_outs]
            metapath_outs = torch.cat(metapath_outs, dim=0)
            h = torch.sum(beta * metapath_outs, dim=0)
            h = self.metapath_attention_fc3[l](h);
            h_ = h;
            
        # output projection layer
        logits = self.out_layer(h);

        # return only the target nodes' logits and embeddings
        return logits, features;
    
    
    
class MAGNN():
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
                 rnn_type='RotatE0',
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
        edge2metapath = {l2_metapaths_list[i]:l2_metapaths_sequence_list[i] for i in range(len(l2_metapaths_sequence_list))};
        metapath2nodetypes = {l2_metapaths_list[i]:l2_metapaths_nodetypes_list[i] for i in range(len(l2_metapaths_list))};
        features_dict = {ntype:torch.FloatTensor(features_dict[ntype]).to(device) for ntype in features_dict}
        in_dims = {ntype:features_dict[ntype].shape[1] for ntype in features_dict}
        block_features_extractor = features_extractor_class(device, features_dict);
        self.net = MAGNN_model(num_layers, target, l2_metapaths_list, edge2metapath, metapath2nodetypes, etypes_list, etypes_complements, in_dims, hidden_dim, attn_vec_dim, num_labels, num_heads, block_features_extractor, rnn_type, dropout_rate);
        
        self.net.to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    def fit(self, train_indices, valid_indices):
        self.net.train()
        dur1 = []
        dur2 = []
        dur3 = []
        train_sampler = MAGNNSampler(self.G, self.metapath_fanout_dict, self.metapath2edge, self.target, self.num_layers);
        loader = DataLoader(dataset=train_indices.tolist(), batch_size=self.batch_size, 
                                collate_fn=train_sampler.sample_blocks, shuffle=True, num_workers=0, worker_init_fn=np.random.seed(self.seed));
        
        # validation sampler
        val_blocks, val_labels = get_block_list_magnn(self.G, valid_indices, self.batch_size, self.target, self.metapath_fanout_dict, self.metapath2edge, self.num_layers, self.labels);
        
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
        test_blocks, test_labels = get_block_list_magnn(self.G, test_indices, self.batch_size, self.target, self.metapath_fanout_dict, self.metapath2edge, self.num_layers, self.labels);
        self.net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(self.save_postfix)))
        test_loss, test_macro_f1, test_micro_f1 = evaluate(self.net, test_blocks, test_labels, self.class_weight, self.device);
        if self.verbose:
            print("Test macro F1: {:.4f} | Test micro F1: {:.4f} | Test Loss: {:.4f}".format(test_macro_f1, test_micro_f1, test_loss))
        return test_loss, test_macro_f1, test_micro_f1;
    
    def predictions(self, test_indices):
        # sampler
        test_blocks, test_labels = get_block_list_magnn(self.G, test_indices, self.batch_size, self.target, self.metapath_fanout_dict, self.metapath2edge, self.num_layers, self.labels);
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
    
class MAGNNSampler:
    def __init__(self, g, metapath2samples, metapath2edge, target, num_layers):
        self.g = g
        self.target = target;
        self.metapath2samples = metapath2samples;
        self.metapath2edge = metapath2edge;
        self.num_layers = num_layers;
        
    def sample_blocks(self, seeds):
        blocks = []
        seeds = torch.LongTensor(seeds);
        cur = seeds
        
        for i in range(self.num_layers):
            rel_dict = {};
            traces_dict = {};
            for metapath in self.metapath2samples:
                traces, types = dgl.sampling.random_walk(self.g, cur.tolist()*self.metapath2samples[metapath], metapath=metapath, restart_prob=0.0);
                src = traces[:,-1];
                src_to_keep = src >= 0;
                traces = traces[src_to_keep];
                src = traces[:,-1]
                dst = traces[:,0]
                rel_dict[(self.target, self.metapath2edge[metapath], self.target)] = (src, dst);
                traces_dict[self.metapath2edge[metapath]] = torch.flip(traces, [1]);
            frontier = dgl.heterograph(rel_dict);
            block = dgl.to_block(frontier, cur, include_dst_in_src=True)
            for metapath in traces_dict:
                eid_indices = block.edges[metapath].data[dgl.NID];
                traces_dict[metapath] = traces_dict[metapath][eid_indices]
            cur = block.srcnodes[self.target].data[dgl.NID]
            blocks.insert(0, (block, traces_dict));
        return seeds, blocks
    
def get_block_list_magnn(G, w_indices, batch_size, target, metapath_fanout_dict, metapath2edge, num_layers, labels):
    w_sampler = MAGNNSampler(G, metapath_fanout_dict, metapath2edge, target, num_layers);
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


