import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.conv import GraphConv

class JKNet(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers=1,
                 mode='cat',
                 dropout=0.):
        super(JKNet, self).__init__()
        
        self.mode = mode
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_dim, hid_dim, activation=F.relu))
        for _ in range(num_layers):
            self.layers.append(GraphConv(hid_dim, hid_dim, activation=F.relu))

        if self.mode == 'cat':
            hid_dim = hid_dim * (num_layers + 1)
        elif self.mode == 'lstm':
            self.lstm = nn.LSTM(hid_dim, (num_layers * hid_dim) // 2, bidirectional=True, batch_first=True)
            self.attn = nn.Linear(2 * ((num_layers * hid_dim) // 2), 1)

        self.output = nn.Linear(hid_dim, out_dim)
        self.reset_params()

    def reset_params(self):
        self.output.reset_parameters()
        for layers in self.layers:
            layers.reset_parameters()
        if self.mode == 'lstm':
            self.lstm.reset_parameters()
            self.attn.reset_parameters()

    def forward(self, g, feats):
        feat_lst = []
        for layer in self.layers:
            feats = self.dropout(layer(g, feats))
            feat_lst.append(feats)
        
        if self.mode == 'cat':
            out = torch.cat(feat_lst, dim=-1)
        elif self.mode == 'max':
            out = torch.stack(feat_lst, dim=-1).max(dim=-1)[0]
        else:
            # lstm
            x = torch.stack(feat_lst, dim=1)
            alpha, _ = self.lstm(x)
            alpha = self.attn(alpha).squeeze(-1)
            alpha = torch.softmax(alpha, dim=-1).unsqueeze(-1)
            out = (x * alpha).sum(dim=1)
        
        g.ndata['h'] = out
        g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))

        return self.output(g.ndata['h'])
