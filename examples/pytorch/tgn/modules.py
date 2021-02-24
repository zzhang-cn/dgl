import torch
import numpy
import dgl
import numpy as np
import torch.nn as nn
from collections import defaultdict
from dgl.base import DGLError
import torch.nn.functional as F
from dgl.ops import edge_softmax
import dgl.function as fn
import copy

class MergeLayer(nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super(MergeLayer,self).__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)


class LinkPredictor(nn.Module):
  def __init__(self, emb_dim, drop=0.3):
    super(LinkPredictor,self).__init__()
    self.fc_1 = torch.nn.Linear(emb_dim, emb_dim)
    self.fc_2 = torch.nn.Linear(emb_dim, emb_dim)
    self.fc_3 = torch.nn.Linear(emb_dim, 1)

  def forward(self, emb_src,emb_dst):
      x = self.fc_1(emb_src) + self.fc_2(emb_dst)
      x = F.relu(x)
      out = self.fc_3(x)
      return out

# Assume the forward pass will 
class MsgLinkPredictor(nn.Module):
    def __init__(self,emb_dim,drop=0.3):
        super(MsgLinkPredictor,self).__init__()
        self.src_fc = nn.Linear(emb_dim,emb_dim)
        self.dst_fc = nn.Linear(emb_dim,emb_dim)
        self.out_fc = nn.Linear(emb_dim,1)

    def link_pred(self,edges):
        src_hid = self.src_fc(edges.src['embedding'])
        dst_hid = self.dst_fc(edges.dst['embedding'])
        score = F.relu(src_hid+dst_hid)
        score = self.out_fc(score)
        return {'score':score}

    def forward(self,x,pos_g,neg_g):
        # Local Scope?
        pos_g.ndata['embedding'] = x
        neg_g.ndata['embedding'] = x

        pos_g.apply_edges(self.link_pred)
        neg_g.apply_edges(self.link_pred)

        pos_escore = pos_g.edata['score']
        neg_escore = neg_g.edata['score']
        return pos_escore,neg_escore

class TimeEncode(nn.Module):
  # Time Encoding proposed by TGAT
  def __init__(self, dimension):
    super(TimeEncode, self).__init__()

    self.dimension = dimension
    self.w = torch.nn.Linear(1, dimension)

    self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                       .double().reshape(dimension, -1))
    self.w.bias = torch.nn.Parameter(torch.zeros(dimension).double())

  def forward(self, t):
    t = t.unsqueeze(dim=2)
    output = torch.cos(self.w(t))
    return output

class MemoryModule(nn.Module):
    def __init__(self,n_node,hidden_dim):
        super(MemoryModule,self).__init__()
        self.n_node = n_node
        self.hidden_dim = hidden_dim
        self.reset_memory()

    def reset_memory(self):
        self.last_update_t = nn.Parameter(torch.zeros(self.n_node).float(),requires_grad=False)
        self.memory        = nn.Parameter(torch.zeros((self.n_node,self.hidden_dim)).float(),requires_grad=False)

    def backup_memory(self):
        return self.memory.clone(), self.last_update_t.clone()

    def restore_memory(self,memory_backup):
        self.memory = memory_backup[0].clone()
        self.last_update_t = memory_backup[1].clone()

    # Which is used for attach to subgraph
    def get_memory(self,node_idxs):
        return self.memory[node_idxs,:]
    
    # When the memory need to be updated
    def set_memory(self,node_idxs,values):
        self.memory[node_idxs,:] = values

    def set_last_update_t(self,node_idxs,values):
        self.last_update_t[node_idxs] = values

    # For safety check
    def get_last_update(self,node_idxs):
        return self.last_update_t[node_idxs]

    def detach_memory(self):
        self.memory.detach_()

class MemoryOperation(nn.Module):
    def __init__(self,updater_type, memory, e_feat_dim, temporal_dim):
        super(MemoryOperation,self).__init__()
        updater_dict = {'gru':nn.GRUCell,'rnn':nn.RNNCell}
        self.memory = memory
        memory_dim = self.memory.hidden_dim
        self.message_dim = memory_dim+memory_dim+e_feat_dim+temporal_dim
        self.updater = updater_dict[updater_type](input_size=self.message_dim,
                                                  hidden_size=memory_dim)
        self.memory = memory
        self.temporal_encoder = TimeEncode(temporal_dim)

    # Here assume g is a subgraph from each iteration
    def stick_feat_to_graph(self,g):
        # How can I ensure order of the node ID
        g.ndata['timestamp'] = self.memory.last_update_t[g.ndata[dgl.NID]]
        g.ndata['memory'] = self.memory.memory[g.ndata[dgl.NID]]

    def msg_fn_cat(self,edges):
        src_delta_time = edges.data['timestamp'] - edges.src['timestamp']
        time_encode = self.temporal_encoder(src_delta_time.unsqueeze(dim=1)).view(len(edges.data['timestamp']),-1)
        ret = torch.cat([edges.src['memory'],edges.dst['memory'],edges.data['feats'],time_encode],dim=1)
        return {'message':ret,'timestamp':edges.data['timestamp']}

    def agg_last(self,nodes):
        timestamp,latest_idx = torch.max(nodes.mailbox['timestamp'],dim=1)
        ret = nodes.mailbox['message'].gather(1,latest_idx.repeat(self.message_dim).view(-1,1,self.message_dim)).view(-1,self.message_dim)
        return {'message_bar':ret.reshape(-1,self.message_dim),'timestamp':timestamp}
    

    def update_memory(self,nodes):
        # It should pass the feature through RNN
        ret = self.updater(nodes.data['message_bar'].float(),nodes.data['memory'].float())
        return {'memory':ret}

    def forward(self,g):
        self.stick_feat_to_graph(g)
        g.update_all(self.msg_fn_cat,self.agg_last,self.update_memory)
        return g
    
class TemporalGATConv(nn.Module):
    def __init__(self,
                 edge_feats,
                 memory_feats,
                 temporal_feats,
                 out_feats,
                 num_heads,
                 allow_zero_in_degree=False):
        super(TemporalGATConv,self).__init__()
        self._edge_feats = edge_feats
        self._memory_feats  = memory_feats
        self._temporal_feats= temporal_feats
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads
        

        self.fc_Q = nn.Linear(self._memory_feats+self._temporal_feats,self._out_feats*self._num_heads,bias=False)
        self.fc_K = nn.Linear(self._memory_feats+self._edge_feats+self._temporal_feats,self._out_feats*self._num_heads,bias=False)
        self.fc_V = nn.Linear(self._memory_feats+self._edge_feats+self._temporal_feats,self._out_feats*self._num_heads,bias=False)
        self.merge= MergeLayer(self._memory_feats,self._out_feats*self._num_heads,512,self._out_feats)
        self.temporal_encoder = TimeEncode(self._temporal_feats)

    def weight_fn(self,edges):
        t0 = torch.zeros_like(edges.dst['timestamp'])
        q = torch.cat([edges.dst['s'],
                       self.temporal_encoder(t0.unsqueeze(dim=1)).view(len(t0),-1)],dim=1)
        time_diff = edges.data['timestamp']-edges.src['timestamp']
        time_encode = self.temporal_encoder(time_diff.unsqueeze(dim=1)).view(len(t0),-1)
        k = torch.cat([edges.src['s'],edges.data['feats'],time_encode],dim=1)
        squeezed_k = self.fc_K(k.float()).view(-1,self._num_heads,self._out_feats)
        squeezed_q = self.fc_Q(q.float()).view(-1,self._num_heads,self._out_feats)
        ret = torch.sum(squeezed_q*squeezed_k,dim=2)
        return {'a':ret,'efeat':squeezed_k} 

    def msg_fn(self,edges):
        ret = edges.data['sa'].view(-1,self._num_heads,1)*edges.data['efeat']
        return {'attn':ret}

    def forward(self,graph,memory,ts):
        graph = graph.local_var() # Using local scope for graph
        if not self._allow_zero_in_degree:
            if(graph.in_degrees()==0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')

        #print("Shape: ",memory.shape,ts.shape)
        graph.srcdata.update({'s':memory,'timestamp':ts})
        graph.dstdata.update({'s':memory,'timestamp':ts})

        # Dot product Calculate the attentio weight
        graph.apply_edges(self.weight_fn)

        # Edge softmax
        graph.edata['sa'] = edge_softmax(graph,graph.edata['a'])/(self._out_feats**0.5)

        # Update dst node Here msg_fn include edge feature
        graph.update_all(self.msg_fn,fn.sum('attn','agg_u'))

        rst = graph.dstdata['agg_u']
        # Implement skip connection
        rst = self.merge(rst.view(-1,self._num_heads*self._out_feats),graph.dstdata['s'])
        return rst