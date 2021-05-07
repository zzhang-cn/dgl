# Copyright 2021 Yifei Ma
# with references from "sklearn.decomposition.LatentDirichletAllocation"
# with the following original authors:
# * Chyi-Kwei Yau (the said scikit-learn implementation)
# * Matthew D. Hoffman (original onlineldavb implementation)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os, functools, warnings
import numpy as np, scipy as sp
import torch
import dgl
from dgl import function as fn


def _lbeta(alpha, axis):
    return torch.lgamma(alpha).sum(axis) - torch.lgamma(alpha.sum(axis))


_sklearn_random_init = torch.distributions.gamma.Gamma(100, 0.01)


def _edge_update(edges, step_size=1):
    """ the Gibbs posterior distribution of z propto theta*beta.
    As step_size -> infty, the result becomes MAP estimate on the dst nodes.
    """
    q = edges.src['weight'] * edges.dst['weight']
    marg = q.sum(axis=1, keepdims=True) + np.finfo(float).eps
    p = q / marg

    return {
        'z': p * step_size,
        'edge_elbo': marg.squeeze(1).log() * step_size,
    }


def _weight_exp(z, ntype, prior):
    """Node weight is approximately normalized for VB along the ntype
    direction.
    """
    prior = prior + z * 0 # convert numpy to torch
    gamma = prior + z

    axis = 1 if ntype == 'doc' else 0 # word
    Elog = torch.digamma(gamma) - torch.digamma(gamma.sum(axis, keepdims=True))

    return Elog.exp()


def _node_update(nodes, prior):
    return {
        'z': nodes.data['z'],
        'weight': _weight_exp(nodes.data['z'], nodes.ntype, prior)
    }


def _update_all(G, ntype, prior, step_size, return_obj=False):
    """Follows Eq (5) of Hoffman et al., 2010.
    """
    G_prop = G.reverse() if ntype == 'doc' else G # word
    msg_fn = lambda edges: _edge_update(edges, step_size)
    node_fn = lambda nodes: _node_update(nodes, prior)

    G_prop.update_all(msg_fn, fn.sum('z','z'), node_fn)

    if return_obj:
        G_prop.update_all(msg_fn, fn.sum('edge_elbo', 'edge_elbo'))

    G.nodes[ntype].data.update(G_prop.nodes[ntype].data)
    return dict(G.nodes[ntype].data)


class LatentDirichletAllocation:
    """LDA model that wors with a HeteroGraph with doc/word node types.
    The model alters the attributes of G arbitrarily,
    but always load word_z if needed.
    This is inspired by [1] and its corresponding scikit-learn implementation.

    References
    ---
    [1] Matthew Hoffman, Francis Bach, David Blei. Online Learning for Latent
    Dirichlet Allocation. Advances in Neural Information Processing Systems 23
    (NIPS 2010).
    [2] Reactive LDA Library blogpost by Yingjie Miao for a similar Gibbs model
    """
    def __init__(
        self, G, n_components,
        prior=None,
        step_size={'doc': 1, 'word': 1}, # use larger value to get MAP
        word_rho=1, # use smaller value for online update
        verbose=True,
        ):
        self.n_components = n_components

        if prior is None:
            prior = {'doc': 1./n_components, 'word': 1./n_components}
        self.prior = prior

        self.step_size = step_size

        self.word_rho = word_rho
        self.word_z = self._load_or_init(G, 'word')['z']
        self.verbose = verbose

    def _load_or_init(self, G, ntype, z=None):
        if z is None:
            z = _sklearn_random_init.sample(
                (G.num_nodes(ntype), self.n_components)
            ).to(G.device)

        G.nodes[ntype].data['z'] = z
        G.apply_nodes(
            lambda nodes: _node_update(nodes, self.prior[ntype]),
            ntype=ntype)
        return dict(G.nodes[ntype].data)

    def _e_step(self, G, reinit_doc=True, mean_change_tol=1e-3, max_iters=100):
        """_e_step implements doc data sampling until convergence or max_iters
        """
        self._load_or_init(G, 'word', self.word_z)

        if reinit_doc or ('weight' not in G.nodes['doc'].data):
            self._load_or_init(G, 'doc')

        for i in range(max_iters):
            doc_z = dict(G.nodes['doc'].data)['z']
            doc_data = _update_all(
                G, 'doc', self.prior['doc'], self.step_size['doc'])
            mean_change = (doc_data['z'] - doc_z).abs().mean()
            if mean_change < mean_change_tol:
                break
        if self.verbose:
            print(f'e-step num_iters={i+1} with mean_change={mean_change:.4f}')
        return doc_data

    transform = _e_step

    def _m_step(self, G):
        """_m_step implements word data sampling and stores word_z stats
        """
        # assume G.nodes['doc'].data has been up to date
        word_data = _update_all(
            G, 'word', self.prior['word'], self.step_size['word'])

        # online update
        self.word_z = (
            (1-self.word_rho) * self.word_z
            +self.word_rho * word_data['z']
        )
        return word_data

    def partial_fit(self, G):
        self._last_word_z = self.word_z
        self._e_step(G)
        self._m_step(G)
        return self

    def fit(self, G, mean_change_tol=1e-3, max_epochs=10):
        for i in range(max_epochs):
            self.partial_fit(G)
            mean_change = (self.word_z - self._last_word_z).abs().mean()
            if self.verbose:
                print(f'epoch {i+1}, '
                      f'perplexity: {self.perplexity(G, False)}, '
                      f'mean_change: {mean_change:.4f}')
            if mean_change < mean_change_tol:
                break
        return self

    def perplexity(self, G, reinit_doc=True):
        """ppl = exp{-sum[log(p(w1,...,wn|d))] / n}
        Follows Eq (15) in Hoffman et al., 2010.
        """
        word_data = self._load_or_init(G, 'word', self.word_z)
        if reinit_doc or ('weight' not in G.nodes['doc'].data):
            self._e_step(G, reinit_doc)
        doc_data = _update_all(
            G, 'doc', self.prior['doc'], self.step_size['doc'],
            return_obj=True)

        # compute E[log p(docs | theta, beta)]
        edge_elbo = (
            doc_data['edge_elbo'].sum() / doc_data['z'].sum()
        ).cpu().numpy()
        if self.verbose:
            print(f'neg_elbo phi: {-edge_elbo:.3f}', end=' ')

        # compute E[log p(theta | alpha) - log q(theta | gamma)]
        doc_elbo = (
            (-doc_data['z'] * doc_data['weight'].log()).sum(axis=1)
            -_lbeta(self.prior['doc'] + doc_data['z'] * 0, axis=1)
            +_lbeta(self.prior['doc'] + doc_data['z'], axis=1)
        )
        doc_elbo = (doc_elbo.sum() / doc_data['z'].sum()).cpu().numpy()
        if self.verbose:
            print(f'theta: {-doc_elbo:.3f}', end=' ')

        # compute E[log p(beta | eta) - log q (beta | lambda)]
        word_elbo = (
            (-word_data['z'] * word_data['weight'].log()).sum(axis=0)
            -_lbeta(self.prior['word'] + word_data['z'] * 0, axis=0)
            +_lbeta(self.prior['word'] + word_data['z'], axis=0)
        )
        word_elbo = (word_elbo.sum() / word_data['z'].sum()).cpu().numpy()
        if self.verbose:
            print(f'beta: {-word_elbo:.3f}')

        return np.exp(-edge_elbo - doc_elbo - word_elbo)


if __name__ == '__main__':
    print('Testing LatentDirichletAllocation via task_example_test.sh ...')
    tf_uv = np.array(np.nonzero(np.random.rand(20,10)<0.5)).T
    G = dgl.heterograph({('doc','topic','word'): tf_uv.tolist()})
    model = LatentDirichletAllocation(G, 5, verbose=False)
    model.fit(G)
    model.transform(G)
    model.perplexity(G)
