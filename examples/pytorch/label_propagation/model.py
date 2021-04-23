import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


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
