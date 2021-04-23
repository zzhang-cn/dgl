import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GATConv

#Semantic attention in the metapath-based aggregation (the same as that in the HAN)
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)

#Metapath-based aggregation (the same as the HANLayer)
class HANLayer(nn.Module):
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []
        #obtain metapath reachable graph
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

#Relational neighbor aggregation
class RelationalAGG(nn.Module):
    def __init__(self, g, in_size, out_size, dropout=0.1):
        super(RelationalAGG, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        #Transform weights for different types of edges
        self.W_T = nn.ModuleDict({
            name : nn.Linear(in_size, out_size, bias = False) for name in g.etypes
        })

        #Attention weights for different types of edges
        self.W_A = nn.ModuleDict({
            name : nn.Linear(out_size, 1, bias = False) for name in g.etypes
        })

        #layernorm
        self.layernorm = nn.LayerNorm(out_size)

        #dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, feat_dict):
        funcs={}
        for srctype, etype, dsttype in g.canonical_etypes:
            g.nodes[dsttype].data['h'] = feat_dict[dsttype] #nodes' original feature
            g.nodes[srctype].data['h'] = feat_dict[srctype]  
            g.nodes[srctype].data['t_h'] = self.W_T[etype](feat_dict[srctype])  #src nodes' transformed feature

            #compute the attention numerator (exp)
            g.apply_edges(fn.u_mul_v('t_h','h','x'),etype=etype)
            g.edges[etype].data['x'] = torch.exp(self.W_A[etype](g.edges[etype].data['x']))

            #first update to compute the attention denominator (\sum exp)
            funcs[etype] = (fn.copy_e('x', 'm'), fn.sum('m', 'att'))  
        g.multi_update_all(funcs, 'sum') 

        funcs={}
        for srctype, etype, dsttype in g.canonical_etypes:
            g.apply_edges(fn.e_div_v('x', 'att', 'att'),etype=etype) #compute attention weights (numerator/denominator)
            funcs[etype] = (fn.u_mul_e('h', 'att', 'm'), fn.sum('m', 'h'))  #\sum(h0*att) -> h1
        #second update to obtain h1
        g.multi_update_all(funcs, 'sum') 

        #apply activation, layernorm, and dropout
        feat_dict={}
        for ntype in g.ntypes:
            feat_dict[ntype] = self.dropout(self.layernorm(F.relu_(g.nodes[ntype].data['h']))) #apply activation, layernorm, and dropout
        
        return feat_dict

class TAHIN(nn.Module):
    def __init__(self, g, meta_paths, in_size, out_size, num_heads, dropout):
        super(TAHIN, self).__init__()

        #embeddings for different types of nodes, h0
        self.initializer = nn.init.xavier_uniform_
        self.feature_dict = nn.ParameterDict({
            ntype: nn.Parameter(self.initializer(torch.empty(g.num_nodes(ntype), in_size))) for ntype in g.ntypes
        })

        #relational neighbor aggregation, this produces h1
        self.RelationalAGG = RelationalAGG(g, in_size, out_size)

        #metapath-based aggregation modules for user and item, this produces h2
        self.meta_paths = meta_paths 
        #one HANLayer for user, one HANLayer for item
        self.hans = nn.ModuleDict({
            key: HANLayer(value, in_size, out_size, num_heads, dropout) for key, value in self.meta_paths.items()
        })

        #layers to combine h0, h1, and h2
        #used to update node embeddings
        self.user_layer1 = nn.Linear((num_heads+1)*out_size, out_size, bias=True)
        self.user_layer2 = nn.Linear(2*out_size, out_size, bias=True)
        self.item_layer1 = nn.Linear((num_heads+1)*out_size, out_size, bias=True)
        self.item_layer2 = nn.Linear(2*out_size, out_size, bias=True)

        #layernorm
        self.layernorm = nn.LayerNorm(out_size)

        #network to score the node pairs
        self.pred = nn.Linear(out_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_size, 1)

    def forward(self, g, user_key, item_key, user_idx, item_idx):
        #relational neighbor aggregation, h1
        h1={}
        h1 = self.RelationalAGG(g, self.feature_dict)

        #metapath-based aggregation, h2
        h2 = {}
        for key in self.meta_paths.keys():
            h2[key] = self.hans[key](g, self.feature_dict[key])

        #update node embeddings
        user_emb = torch.cat((h1[user_key], h2[user_key]), 1)
        item_emb = torch.cat((h1[item_key], h2[item_key]), 1)
        user_emb = self.user_layer1(user_emb)
        item_emb = self.item_layer1(item_emb)
        user_emb = self.user_layer2(torch.cat((user_emb, self.feature_dict[user_key]), 1))
        item_emb = self.item_layer2(torch.cat((item_emb, self.feature_dict[item_key]), 1))

        #Relu
        user_emb = F.relu_(user_emb)
        item_emb = F.relu_(item_emb)
        
        #layer norm
        user_emb = self.layernorm(user_emb)
        item_emb = self.layernorm(item_emb)
        
        #obtain users/items embeddings and their interactions
        user_feat = user_emb[user_idx]
        item_feat = item_emb[item_idx]
        interaction = user_feat*item_feat

        #score the node pairs
        pred = self.pred(interaction)
        pred = self.dropout(pred) #dropout
        pred = self.fc(pred)
        pred = torch.sigmoid(pred)

        return pred.squeeze(1)
        