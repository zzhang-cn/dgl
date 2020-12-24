"""Utilities for pytorch NN package"""
#pylint: disable=no-member, invalid-name

import jax
from jax import numpy as jnp
import flax
from flax import linen as nn
from ... import DGLGraph
from ...base import dgl_warning

def matmul_maybe_select(A, B):
    """Perform Matrix multiplication C = A * B but A could be an integer id vector.

    If A is an integer vector, we treat it as multiplying a one-hot encoded tensor.
    In this case, the expensive dense matrix multiply can be replaced by a much
    cheaper index lookup.

    For example,
    ::

        A = [2, 0, 1],
        B = [[0.1, 0.2],
             [0.3, 0.4],
             [0.5, 0.6]]

    then matmul_maybe_select(A, B) is equivalent to
    ::

        [[0, 0, 1],     [[0.1, 0.2],
         [1, 0, 0],  *   [0.3, 0.4],
         [0, 1, 0]]      [0.5, 0.6]]

    In all other cases, perform a normal matmul.

    Parameters
    ----------
    A : torch.Tensor
        lhs tensor
    B : torch.Tensor
        rhs tensor

    Returns
    -------
    C : torch.Tensor
        result tensor
    """
    if A.dtype == jnp.int64 and len(A.shape) == 1:
        return jnp.take(B, A, 0)
    else:
        return jnp.matmul(A, B)

def bmm_maybe_select(A, B, index):
    """Slice submatrices of A by the given index and perform bmm.

    B is a 3D tensor of shape (N, D1, D2), which can be viewed as a stack of
    N matrices of shape (D1, D2). The input index is an integer vector of length M.
    A could be either:
    (1) a dense tensor of shape (M, D1),
    (2) an integer vector of length M.
    The result C is a 2D matrix of shape (M, D2)

    For case (1), C is computed by bmm:
    ::

        C[i, :] = matmul(A[i, :], B[index[i], :, :])

    For case (2), C is computed by index select:
    ::

        C[i, :] = B[index[i], A[i], :]

    Parameters
    ----------
    A : torch.Tensor
        lhs tensor
    B : torch.Tensor
        rhs tensor
    index : torch.Tensor
        index tensor

    Returns
    -------
    C : torch.Tensor
        return tensor
    """
    if A.dtype == jnp.int64 and len(A.shape) == 1:
        # following is a faster version of B[index, A, :]
        B = B.reshape((-1, B.shape[2]))
        flatidx = index * B.shape[1] + A
        return jnp.take(B, flatidx, 0)

    else:
        BB = jnp.take(B, index, 0)
        return jax.lax.batch_matmul(jnp.expand_dims(A, 1), BB).squeeze()

# pylint: disable=W0235
class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x

class WeightBasis(nn.Module):
    r"""Basis decomposition module.

    Basis decomposition is introduced in "`Modeling Relational Data with Graph
    Convolutional Networks <https://arxiv.org/abs/1703.06103>`__"
    and can be described as below:

    .. math::

        W_o = \sum_{b=1}^B a_{ob} V_b

    Each weight output :math:`W_o` is essentially a linear combination of basis
    transformations :math:`V_b` with coefficients :math:`a_{ob}`.

    If is useful as a form of regularization on a large parameter matrix. Thus,
    the number of weight outputs is usually larger than the number of bases.

    Parameters
    ----------
    shape : tuple[int]
        Shape of the basis parameter.
    num_bases : int
        Number of bases.
    num_outputs : int
        Number of outputs.
    """
    def __init__(self,
                 shape,
                 num_bases,
                 num_outputs):
        super(WeightBasis, self).__init__()
        self.shape = shape
        self.num_bases = num_bases
        self.num_outputs = num_outputs

        if num_outputs <= num_bases:
            dgl_warning('The number of weight outputs should be larger than the number'
                        ' of bases.')

        self.weight = nn.Parameter(jnp.Tensor(self.num_bases, *shape))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        # linear combination coefficients
        self.w_comp = nn.Parameter(jnp.Tensor(self.num_outputs, self.num_bases))
        nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

    def forward(self):
        r"""Forward computation

        Returns
        -------
        weight : torch.Tensor
            Composed weight tensor of shape ``(num_outputs,) + shape``
        """
        # generate all weights from bases
        weight = jnp.matmul(self.w_comp, self.weight.view(self.num_bases, -1))
        return weight.view(self.num_outputs, *self.shape)
