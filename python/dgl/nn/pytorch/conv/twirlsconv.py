"""Torch modules for TWIRLS"""
# pylint: disable=invalid-name, useless-super-delegation, no-member

import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from .... import function as fn


class TWIRLSConv(nn.Module):
    r"""

    Description
    -----------
    Together with iteratively reweighting least squre from paper `Graph Neural Networks Inspired
    by Classical Iterative Algorithms <https://arxiv.org/pdf/2103.06064.pdf>`__.

    Parameters
    ----------
    input_d : int
        Number of input features.
    output_d : int
        Number of output features.
    hidden_d : int
        Size of hidden layers.
    prop_step : int
        Number of propagation steps
    num_mlp_before : int
        Number of mlp layers before propagation. Default: ``1``.
    num_mlp_after : int
        Number of mlp layers after propagation.  Default: ``1``.
    norm : str
        The type of norm layers inside mlp layers. Can be ``'batch'``, ``'layer'`` or ``'none'``.
        Default: ``'none'``
    precond : str
        If True, use pre conditioning and unormalized laplacian, else not use pre conditioning
        and use normalized laplacian. Default: ``True``
    alp : float
        The :math:`\alpha` in paper. If equal to :math:`0`, will be automatically decided based
        on other hyper prameters. Default: ``0``.
    lam : float
        The :math:`\lambda` in paper. Default: ``1``.
    attention : bool
        If ``True``, add an attention layer inside propagations. Default: ``False``.
    tau : float
        The :math:`\tau` in paper. Default: ``0.2``.
    T : float
        The :math:`T` in paper. If < 0, :math:`T` will be set to `\infty`. Default: ``-1``.
    p : float
        The :math:`p` in paper. Default: ``1``.
    use_eta : bool
        If ``True``, add a learnable weight on each dimension in attention. Default: ``False``.
    attn_bef : bool
        If ``True``, add another attention layer before propagation. Default: ``False``.
    dropout : float
        The dropout rate in mlp layers. Default: ``0.0``.
    attn_dropout : float
        The dropout rate of attention values. Default: ``0.0``.
    inp_dropout : float
        The dropout rate on input features. Default: ``0.0``.


    Note
    ----
     ``add_self_loop`` will be automatically called before propagation.

    Example
    -------
    >>> import dgl
    >>> from dgl.nn import TWIRLSConv
    >>> import torch as th

    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = TWIRLSConv(10, 2, 128, prop_step = 64)
    >>> res = conv(g , feat)
    >>> res
    tensor([[ 0.4556, -2.6692],
            [ 0.4556, -2.6692],
            [ 0.4556, -2.6692],
            [ 1.0112, -5.9241],
            [ 0.8011, -4.6935],
            [ 0.8844, -5.1814]], grad_fn=<AddmmBackward>)
    """

    def __init__(self,
                 input_d,
                 output_d,
                 hidden_d,
                 prop_step,
                 num_mlp_before=1,
                 num_mlp_after=1,
                 norm='none',
                 precond=True,
                 alp=0,
                 lam=1,
                 attention=False,
                 tau=0.2,
                 T=-1,
                 p=1,
                 use_eta=False,
                 attn_bef=False,
                 dropout=0.0,
                 attn_dropout=0.0,
                 inp_dropout=0.0,
                 ):

        super().__init__()
        self.input_d = input_d
        self.output_d = output_d
        self.hidden_d = hidden_d
        self.prop_step = prop_step
        self.num_mlp_before = num_mlp_before
        self.num_mlp_after = num_mlp_after
        self.norm = norm
        self.precond = precond
        self.attention = attention
        self.alp = alp
        self.lam = lam
        self.tau = tau
        self.T = T
        self.p = p
        self.use_eta = use_eta
        self.init_att = attn_bef
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.inp_dropout = inp_dropout

        # ----- initialization of some variables -----
        # where to put attention
        self.attn_aft = prop_step // 2 if attention else -1

        # whether we can cache unfolding result
        self.cacheable = (
            not self.attention) and self.num_mlp_before == 0 and self.inp_dropout <= 0
        if self.cacheable:
            self.cached_unfolding = None

        # if only one layer, then no hidden size
        self.size_bef_unf = self.hidden_d
        self.size_aft_unf = self.hidden_d
        if self.num_mlp_before == 0:
            self.size_aft_unf = self.input_d  # as the input  of mlp_aft
        if self.num_mlp_after == 0:
            self.size_bef_unf = self.output_d  # as the output of mlp_bef

        # ----- computational modules -----
        self.mlp_bef = MLP(self.input_d, self.hidden_d, self.size_bef_unf, self.num_mlp_before,
                           self.dropout, self.norm, init_activate=False)

        self.unfolding = UnfoldingAndAttention(self.hidden_d, self.alp, self.lam, self.prop_step,
                                               self.attn_aft, self.tau, self.T, self.p,
                                               self.use_eta, self.init_att, self.attn_dropout,
                                               self.precond)

        # if there are really transformations before unfolding, then do init_activate in mlp_aft
        self.mlp_aft = MLP(self.size_aft_unf, self.hidden_d, self.output_d, self.num_mlp_after,
                           self.dropout, self.norm,
                           init_activate=(self.num_mlp_before > 0) and (
                               self.num_mlp_after > 0)
                           )

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Run TWIRLS forward.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The initial node features.
        Returns
        -------
        torch.Tensor
            The output feature

        Note
        ----
        * Input shape: :math:`(N, \text{input_d})` where :math:`N` is the number of nodes.
        * Output shape: :math:`(N, \text{output_d})`.
        """

        # ensure self loop
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()

        x = feat

        if self.cacheable:
            # to cache unfolding result becase there is no paramaters before it
            if self.cached_unfolding is None:
                self.cached_unfolding = self.unfolding(graph, x)

            x = self.cached_unfolding
        else:
            if self.inp_dropout > 0:
                x = F.dropout(x, self.inp_dropout, training=self.training)
            x = self.mlp_bef(x)
            x = self.unfolding(graph, x)

        x = self.mlp_aft(x)

        return x


class Propagate(nn.Module):
    r"""

    Description
    -----------
    The propagation method which is with pre-conditioning and reparameterizing. Correspond to
    eq.28 in the paper.

    """

    def __init__(self):
        super().__init__()

    def _prop(self, graph, Y, lam):
        """propagation part.
        """
        Y = D_power_bias_X(graph, Y, -0.5, lam, 1 - lam)
        Y = AX(graph, Y)
        Y = D_power_bias_X(graph, Y, -0.5, lam, 1 - lam)

        return Y

    def forward(self, graph, Y, X, alp, lam):
        r"""

        Description
        -----------
        Propagation forward.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        Y : torch.Tensor
            The feature under propagation. Corresponds to :math:`Z^{(k)}` in eq.28 in the paper.
        X : torch.Tensor
            The original feature. Corresponds to :math:`Z^{(0)}` in eq.28 in the paper.
        alp : float
            The step size. Corresponds to :math:`\alpha` in the paper.
        lam : torch.Tensor
            The coefficient of smoothing term. Corresponds to :math:`\lambda` in the paper.
        Returns
        -------
        torch.Tensor
            Propagated feature. :math:`Z^{(k+1)}` in eq.28 in the paper.
        """

        return (1 - alp) * Y + alp * lam * self._prop(graph, Y, lam) \
            + alp * D_power_bias_X(graph, X, -1, lam, 1 - lam)


class PropagateNoPrecond(nn.Module):
    r"""

    Description
    -----------
    The propagation method which is without pre-conditioning and reparameterizing and using
    normalized laplacian.
    Correspond to eq.30 in the paper.
    """

    def __init__(self):
        super().__init__()

    def forward(self, graph, Y, X, alp, lam):
        r"""

        Description
        -----------
        Propagation forward.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        Y : torch.Tensor
            The feature under propagation. Corresponds to :math:`Y^{(k)}` in eq.30 in the paper.
        X : torch.Tensor
            The original feature. Corresponds to :math:`Y^{(0)}` in eq.30 in the paper.
        alp : float
            The step size. Corresponds to :math:`\alpha` in the paper.
        lam : torch.Tensor
            The coefficient of smoothing term. Corresponds to :math:`\lambda` in the paper.
        Returns
        -------
        torch.Tensor
            Propagated feature. :math:`Y^{(k+1)}` in eq.30 in the paper.
        """

        return (1 - alp * lam - alp) * Y + alp * lam * normalized_AX(graph, Y) + alp * X


class Attention(nn.Module):
    r"""

    Description
    -----------
    The attention function. Correspond to :math:`s` in eq.27 the paper.

    Parameters
    ----------
    tau : float
        The lower thresholding parameter. Correspond to :math:`\tau` in the paper.
    T : float
        The upper thresholding parameter. Correspond to :math:`T` in the paper.
    p : float
        Correspond to :math:`\rho` in the paper..
    attn_dropout : float
        the dropout rate of attention value. Default: ``0.0``.

    Returns
    -------
    torch.Tensor
        The output feature
    """

    def __init__(self, tau, T, p, attn_dropout=0.0):
        super().__init__()

        self.tau = tau
        self.T = T
        self.p = p
        self.attn_dropout = attn_dropout

    def reweighting(self, graph):
        """Compute graph edge weight. Would be stored in ``graph.edata['w']``"""

        w = graph.edata["w"]

        # It is not activation here but to ensure w > 0.
        # w can be < 0 here because of some precision issue in dgl, which causes NaN afterwards.
        w = F.relu(w) + 1e-7

        w = tc.pow(w, 1 - 0.5 * self.p)

        w[(w < self.tau)] = self.tau
        if self.T > 0:
            w[(w > self.T)] = float("inf")

        w = 1 / w

        # if not (w == w).all():
        #     raise "nan occured!"

        graph.edata["w"] = w + 1e-9  # avoid 0 degree

    def forward(self, graph, Y, etas=None):
        r"""

        Description
        -----------
        Attention forward. Will update ``graph.edata['w']`` and ``graph.ndata['deg']``.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        Y : torch.Tensor
            The feature to compute attention.
        etas : float
            The weight of each dimension. If ``None``, then weight of each dimension is 1.
            Default: ``None``.

        Returns
        -------
        DGLGraph
            The graph.
        """

        if etas is not None:
            Y = Y * etas.view(-1)

        # computing edge distance
        graph.srcdata["h"] = Y
        graph.srcdata["h_norm"] = (Y ** 2).sum(-1)
        graph.apply_edges(fn.u_dot_v("h", "h", "dot_"))
        graph.apply_edges(fn.u_add_v("h_norm", "h_norm", "norm_"))
        graph.edata["dot_"] = graph.edata["dot_"].view(-1)
        graph.edata["norm_"] = graph.edata["norm_"].view(-1)
        graph.edata["w"] = graph.edata["norm_"] - 2 * graph.edata["dot_"]

        # apply edge distance to get edge weight
        self.reweighting(graph)

        # update node degrees
        graph.update_all(fn.copy_e("w", "m"), fn.sum("m", "deg"))
        graph.ndata["deg"] = graph.ndata["deg"].view(-1)

        # attention dropout. the implementation can ensure the degrees do not change in expectation.
        # FIXME: consider if there is a better way
        if self.attn_dropout > 0:
            graph.edata["w"] = F.dropout(
                graph.edata["w"], self.attn_dropout, training=self.training)

        return graph


def normalized_AX(graph, X):
    """Y = D^{-1/2}AD^{-1/2}X"""

    Y = D_power_X(graph, X, -0.5)  # Y = D^{-1/2}X
    Y = AX(graph, Y)  # Y = AD^{-1/2}X
    Y = D_power_X(graph, Y, -0.5)  # Y = D^{-1/2}AD^{-1/2}X

    return Y


def AX(graph, X):
    """Y = AX"""

    graph.srcdata["h"] = X
    graph.update_all(
        fn.u_mul_e("h", "w", "m"), fn.sum("m", "h"),
    )
    Y = graph.dstdata["h"]

    return Y


def D_power_X(graph, X, power):
    """Y = D^{power}X"""

    degs = graph.ndata["deg"]
    norm = tc.pow(degs, power)
    Y = X * norm.view(X.size(0), 1)
    return Y


def D_power_bias_X(graph, X, power, coeff, bias):
    """Y = (coeff*D + bias*I)^{power} X"""
    degs = graph.ndata["deg"]
    degs = coeff * degs + bias
    norm = tc.pow(degs, power)
    Y = X * norm.view(X.size(0), 1)
    return Y


class UnfoldingAndAttention(nn.Module):
    r"""

    Description
    -----------
    Combine propagation and attention together.

    Parameters
    ----------
    d : int
        Size of graph feature.
    alp : float
        Step size. :math:`\alpha` in ther paper.
    lam : int
        Coefficient of graph smooth term. :math:`\lambda` in ther paper.
    prop_step : int
        Number of propagation steps
    attn_aft : int
        Where to put attention layer. i.e. number of propagation steps before attention.
        If set to ``-1``, then no attention.
    tau : float
        The lower thresholding parameter. Correspond to :math:`\tau` in the paper.
    T : float
        The upper thresholding parameter. Correspond to :math:`T` in the paper.
    p : float
        Correspond to :math:`\rho` in the paper..
    use_eta : bool
        If `True`, learn a weight vector for each dimension when doing attention.
    init_att : bool
        If ``True``, add an extra attention layer before propagation.
    attn_dropout : float
        the dropout rate of attention value. Default: ``0.0``.
    precond : bool
        If ``True``, use pre-conditioned & reparameterized version propagation (eq.28), else use
        normalized laplacian (eq.30).

    Example
    -------
    >>> import dgl
    >>> from dgl.nn import TWIRLSUnfoldingAndAttention
    >>> import torch as th

    >>> g = dgl.graph(([0, 1, 2, 3, 2, 5], [1, 2, 3, 4, 0, 3])).add_self_loop()
    >>> feat = th.ones(6,5)
    >>> prop = TWIRLSUnfoldingAndAttention(10, 1, 1, prop_step=3)
    >>> res = prop(g,feat)
    >>> res
    tensor([[2.5000, 2.5000, 2.5000, 2.5000, 2.5000],
            [2.5000, 2.5000, 2.5000, 2.5000, 2.5000],
            [2.5000, 2.5000, 2.5000, 2.5000, 2.5000],
            [3.7656, 3.7656, 3.7656, 3.7656, 3.7656],
            [2.5217, 2.5217, 2.5217, 2.5217, 2.5217],
            [4.0000, 4.0000, 4.0000, 4.0000, 4.0000]])

    """

    def __init__(self,
                 d,
                 alp,
                 lam,
                 prop_step,
                 attn_aft=-1,
                 tau=0.2,
                 T=-1,
                 p=1,
                 use_eta=False,
                 init_att=False,
                 attn_dropout=0,
                 precond=True,
                 ):

        super().__init__()

        self.d = d
        self.alp = alp if alp > 0 else 1 / (lam + 1)  # automatic set alpha
        self.lam = lam
        self.tau = tau
        self.p = p
        self.prop_step = prop_step
        self.attn_aft = attn_aft
        self.use_eta = use_eta
        self.init_att = init_att

        prop_method = Propagate if precond else PropagateNoPrecond
        self.prop_layers = nn.ModuleList(
            [prop_method() for _ in range(prop_step)])

        self.init_attn = Attention(
            tau, T, p, attn_dropout) if self.init_att else None
        self.attn_layer = Attention(
            tau, T, p, attn_dropout) if self.attn_aft >= 0 else None
        self.etas = nn.Parameter(tc.ones(d)) if self.use_eta else None

    def forward(self, g, X):
        r"""

        Description
        -----------
        Compute forward pass of propagation & attention.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        X : torch.Tensor
            Init features.

        Returns
        -------
        torch.Tensor
            The graph.
        """
        Y = X

        g.edata["w"] = tc.ones(g.number_of_edges(), 1, device=g.device)
        g.ndata["deg"] = g.in_degrees().float()

        if self.init_att:
            g = self.init_attn(g, Y, self.etas)

        for k, layer in enumerate(self.prop_layers):

            # do unfolding
            Y = layer(g, Y, X, self.alp, self.lam)

            # do attention at certain layer
            if k == self.attn_aft - 1:
                g = self.attn_layer(g, Y, self.etas)

        return Y


class MLP(nn.Module):
    r"""

    Description
    -----------
    An MLP module.

    Parameters
    ----------
    input_d : int
        Number of input features.
    output_d : int
        Number of output features.
    hidden_d : int
        Size of hidden layers.
    num_layers : int
        Number of mlp layers.
    dropout : float
        The dropout rate in mlp layers.
    norm : str
        The type of norm layers inside mlp layers. Can be ``'batch'``, ``'layer'`` or ``'none'``.
    init_activate : bool
        If add a relu at the beginning.

    """

    def __init__(self, input_d, hidden_d, output_d, num_layers, dropout, norm, init_activate):
        super().__init__()

        self.init_activate = init_activate
        self.norm = norm
        self.dropout = dropout

        self.layers = nn.ModuleList([])

        if num_layers == 1:
            self.layers.append(nn.Linear(input_d, output_d))
        elif num_layers > 1:
            self.layers.append(nn.Linear(input_d, hidden_d))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_d, hidden_d))
            self.layers.append(nn.Linear(hidden_d, output_d))

        # how many norm layers we have
        self.norm_cnt = num_layers-1+int(init_activate)
        if norm == "batch":
            self.norms = nn.ModuleList(
                [nn.BatchNorm1d(hidden_d) for _ in range(self.norm_cnt)])
        elif norm == "layer":
            self.norms = nn.ModuleList(
                [nn.LayerNorm(hidden_d) for _ in range(self.norm_cnt)])

        self.reset_params()

    def reset_params(self):
        """reset mlp parameters using xavier_norm"""
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight.data)
            nn.init.constant_(layer.bias.data, 0)

    def activate(self, x):
        """do normlaization and activation"""
        if self.norm != "none":
            x = self.norms[self.cur_norm_idx](x)  # use the last norm layer
            self.cur_norm_idx += 1
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

    def forward(self, x):
        """The forward pass of mlp."""
        self.cur_norm_idx = 0

        if self.init_activate:
            x = self.activate(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:  # do not activate in the last layer
                x = self.activate(x)

        return x
