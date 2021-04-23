import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from dgl.nn.pytorch.conv import GATConv


class MLPLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPLinear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, dropout=0.):
        super(MLP, self).__init__()
        assert num_layers >= 2

        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.linears.append(nn.Linear(in_dim, hid_dim))
        self.bns.append(nn.BatchNorm1d(hid_dim))

        for _ in range(num_layers - 2):
            self.linears.append(nn.Linear(hid_dim, hid_dim))
            self.bns.append(nn.BatchNorm1d(hid_dim))
        
        self.linears.append(nn.Linear(hid_dim, out_dim))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.linears:
            layer.reset_parameters()
        for layer in self.bns:
            layer.reset_parameters()

    def forward(self, x):
        for linear, bn in zip(self.linears[:-1], self.bns):
            x = bn(linear(x))
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linears[-1](x)
        return F.log_softmax(x, dim=-1)


class Bias(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.bias = nn.Parameter(torch.Tensor(size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return x + self.bias


class GAT(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes,
                 n_hidden,
                 n_layers,
                 n_heads,
                 activation,
                 dropout=0.0,
                 attn_drop=0.0):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.num_heads = n_heads

        self.convs = nn.ModuleList()
        self.linear = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.biases = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            out_channels = n_heads

            self.convs.append(GATConv(in_hidden,
                                      out_hidden,
                                      num_heads=n_heads,
                                      attn_drop=attn_drop))
            self.linear.append(nn.Linear(in_hidden, out_channels * out_hidden, bias=False))
            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_channels * out_hidden))

        self.bias_last = Bias(n_classes)

        self.dropout0 = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.dropout0(h)

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)
            linear = self.linear[i](h).view(conv.shape)

            h = conv + linear

            if i < self.n_layers - 1:
                h = h.flatten(1)
                h = self.bns[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        h = h.mean(1)
        h = self.bias_last(h)

        return F.log_softmax(h, dim=-1)

class LabelPropagation(nn.Module):
    r"""

    Description
    -----------
    Introduced in `Learning from Labeled and Unlabeled Datawith Label Propagation <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.3864&rep=rep1&type=pdf>`_

    .. math::
        \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \alpha) \mathbf{Y},

    where unlabeled data is inferred by labeled data via propagation.

    Parameters
    ----------
        num_layers: int
            The number of propagations.
        alpha: float
            The :math:`\alpha` coefficient.
    """
    def __init__(self, num_layers, alpha):
        super(LabelPropagation, self).__init__()

        self.num_layers = num_layers
        self.alpha = alpha
    
    @torch.no_grad()
    def forward(self, g, labels, mask=None, post_step=lambda y: y.clamp_(0., 1.)):
        with g.local_scope():
            if labels.dtype == torch.long:
                labels = F.one_hot(labels.view(-1)).to(torch.float32)
            
            y = labels
            if mask is not None:
                y = torch.zeros_like(labels)
                y[mask] = labels[mask]
            
            last = (1 - self.alpha) * y

            for _ in range(self.num_layers):
                degs = g.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5).to(labels.device).unsqueeze(1)
                g.ndata['h'] = y * norm
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                y = last + self.alpha * g.ndata.pop('h') * norm
                y = post_step(y)
            
            return y


class CorrectAndSmooth(nn.Module):
    r"""

    Description
    -----------
    Introduced in `Combining Label Propagation and Simple Models Out-performs Graph Neural Networks <https://arxiv.org/abs/2010.13993>`_

    Parameters
    ----------
        num_correction_layers: int
            The number of correct propagations.
        correction_alpha: float
            The coefficient of correction.
        num_smoothing_layers: int
            The number of smooth propagations.
        smoothing_alpha: float
            The coefficient of smoothing.
        autoscale: bool, optional
            If set to True, will automatically determine the scaling factor :math:`\sigma`. Default is True.
        scale: float, optional
            The scaling factor :math:`\sigma`, in case :obj:`autoscale = False`. Default is 1.
    """
    def __init__(self,
                 num_correction_layers,
                 correction_alpha,
                 num_smoothing_layers,
                 smoothing_alpha,
                 autoscale=True,
                 scale=1.):
        super(CorrectAndSmooth, self).__init__()
        
        self.autoscale = autoscale
        self.scale = scale

        self.prop1 = LabelPropagation(num_correction_layers, correction_alpha)
        self.prop2 = LabelPropagation(num_smoothing_layers, smoothing_alpha)

    def correct(self, g, y_soft, y_true, mask):
        with g.local_scope():
            assert abs(float(y_soft.sum()) / y_soft.size(0) - 1.0) < 1e-2
            numel = int(mask.sum()) if mask.dtype == torch.bool else mask.size(0)
            assert y_true.size(0) == numel

            if y_true.dtype == torch.long:
                y_true = F.one_hot(y_true.view(-1), y_soft.size(-1)).to(y_soft.dtype)
            
            error = torch.zeros_like(y_soft)
            error[mask] = y_true - y_soft[mask]

            if self.autoscale:
                smoothed_error = self.prop1(g, error, post_step=lambda x: x.clamp_(-1., 1.))
                sigma = error[mask].abs().sum() / numel
                scale = sigma / smoothed_error.abs().sum(dim=1, keepdim=True)
                scale[scale.isinf() | (scale > 1000)] = 1.0
                return y_soft + scale * smoothed_error
            else:
                def fix_input(x):
                    x[mask] = error[mask]
                    return x
                
                smoothed_error = self.prop1(g, error, post_step=fix_input)
                return y_soft + self.scale * smoothed_error

    def smooth(self, g, y_soft, y_true, mask):
        with g.local_scope():
            numel = int(mask.sum()) if mask.dtype == torch.bool else mask.size(0)
            assert y_true.size(0) == numel

            if y_true.dtype == torch.long:
                y_true = F.one_hot(y_true.view(-1), y_soft.size(-1)).to(y_soft.dtype)
            
            y_soft[mask] = y_true
            return self.prop2(g, y_soft)
