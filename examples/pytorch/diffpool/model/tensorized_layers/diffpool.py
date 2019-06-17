import torch
from torch import nn as nn

from model.tensorized_layers.assignment import DiffPoolAssignment
from model.tensorized_layers.graphsage import BatchedGraphSAGE


# from model.layers.loss.link_pred import LinkPredLoss


class EntropyLoss(nn.Module):
    # Return Scalar
    def forward(self, adj, anext, s_l):
        entropy = (torch.distributions.Categorical(probs=s_l).entropy()).sum(-1).mean(-1)
        assert not torch.isnan(entropy)
        return entropy


class LinkPredLoss(nn.Module):

    def forward(self, adj, anext, s_l):
        link_pred_loss = (adj - s_l.matmul(s_l.transpose(-1, -2))).norm(dim=(1, 2))
        link_pred_loss = link_pred_loss / (adj.size(1) * adj.size(2))
        return link_pred_loss.mean()

class BatchedDiffPool(nn.Module):
    def __init__(self, nfeat, nnext, nhid, link_pred=True, entropy=False):
        super(BatchedDiffPool, self).__init__()
        self.link_pred = link_pred
        self.log = {}
        self.min_cut = True
        self.link_pred_layer = LinkPredLoss()
        self.embed = BatchedGraphSAGE(nfeat, nhid, use_bn=True)
        self.assign = DiffPoolAssignment(nfeat, nnext)
        self.reg_loss = nn.ModuleList([])
        self.loss_log = {}
        if link_pred:
            self.reg_loss.append(LinkPredLoss())
        if entropy:
            self.reg_loss.append(EntropyLoss())


    def forward(self, x, adj, log=False):
        z_l = self.embed(x, adj)
        s_l = self.assign(x, adj)
        if log:
            self.log['s'] = s_l.cpu().numpy()
        xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
        anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)

        for loss_layer in self.reg_loss:
            loss_name = str(type(loss_layer).__name__)
            self.loss_log[loss_name] = loss_layer(adj, anext, s_l)
        if log:
            self.log['a'] = anext.cpu().numpy()
        return xnext, anext


