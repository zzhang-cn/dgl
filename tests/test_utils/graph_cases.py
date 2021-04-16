from collections import defaultdict
import backend as F
import dgl
import numpy as np
import networkx as nx
import numpy as np
import scipy.sparse as ssp
import backend as F

case_registry = defaultdict(list)

def register_case(labels):
    def wrapper(fn):
        for lbl in labels:
            case_registry[lbl].append(fn)
        fn.__labels__ = labels
        return fn
    return wrapper

def get_cases(labels=None, exclude=[]):
    """Get all graph instances of the given labels."""
    cases = set()
    if labels is None:
        # get all the cases
        labels = case_registry.keys()
    for lbl in labels:
        for case in case_registry[lbl]:
            if not any([l in exclude for l in case.__labels__]):
                cases.add(case)
    return [fn() for fn in cases]

@register_case(['bipartite', 'zero-degree'])
def bipartite1():
    return dgl.heterograph({('_U', '_E', '_V'): ([0, 0, 0, 2, 2, 3],
                                                 [0, 1, 4, 1, 4, 3])})

@register_case(['bipartite'])
def bipartite_full():
    return dgl.heterograph({('_U', '_E', '_V'): ([0, 0, 0, 0, 1, 1, 1, 1],
                                                 [0, 1, 2, 3, 0, 1, 2, 3])})

@register_case(['homo'])
def graph0():
    return dgl.graph(([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 6, 6, 7, 8, 9],
                      [4, 5, 1, 2, 4, 7, 9, 8 ,6, 4, 1, 0, 1, 0, 2, 3, 5]))

@register_case(['homo', 'zero-degree', 'homo-zero-degree'])
def bipartite1():
    return dgl.graph(([0, 0, 0, 2, 2, 3], [0, 1, 4, 1, 4, 3]))

@register_case(['homo', 'has_feature'])
def graph1():
    g = dgl.graph(([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 6, 6, 7, 8, 9],
                   [4, 5, 1, 2, 4, 7, 9, 8 ,6, 4, 1, 0, 1, 0, 2, 3, 5]), device=F.cpu())
    g.ndata['h'] = F.copy_to(F.randn((g.number_of_nodes(), 2)), F.cpu())
    g.edata['w'] = F.copy_to(F.randn((g.number_of_edges(), 3)), F.cpu())
    return g

@register_case(['homo', 'has_scalar_e_feature'])
def graph1():
    g = dgl.graph(([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 6, 6, 7, 8, 9],
                   [4, 5, 1, 2, 4, 7, 9, 8 ,6, 4, 1, 0, 1, 0, 2, 3, 5]), device=F.cpu())
    g.ndata['h'] = F.copy_to(F.randn((g.number_of_nodes(), 2)), F.cpu())
    g.edata['scalar_w'] = F.copy_to(F.abs(F.randn((g.number_of_edges(),))), F.cpu())
    return g

@register_case(['homo', 'row_sorted'])
def graph2():
    return dgl.graph(([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 6, 6, 7, 8, 9],
                      [4, 5, 1, 2, 4, 7, 9, 8 ,6, 4, 1, 0, 1, 0, 2, 3, 5]),
                      row_sorted=True)

@register_case(['homo', 'row_sorted', 'col_sorted'])
def graph3():
    return dgl.graph(([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 6, 6, 7, 8, 9],
                      [1, 4, 5, 2, 4, 7, 8, 9 ,1, 4, 6, 0, 0, 1, 2, 3, 5]),
                      row_sorted=True, col_sorted=True)

@register_case(['hetero', 'has_feature'])
def heterograph0():
    g = dgl.heterograph({
        ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 1, 1]),
        ('developer', 'develops', 'game'): ([0, 1], [0, 1])}, device=F.cpu())
    g.nodes['user'].data['h'] = F.copy_to(F.randn((g.number_of_nodes('user'), 3)), F.cpu())
    g.nodes['game'].data['h'] = F.copy_to(F.randn((g.number_of_nodes('game'), 2)), F.cpu())
    g.nodes['developer'].data['h'] = F.copy_to(F.randn((g.number_of_nodes('developer'), 3)), F.cpu())
    g.edges['plays'].data['h'] = F.copy_to(F.randn((g.number_of_edges('plays'), 1)), F.cpu())
    g.edges['develops'].data['h'] = F.copy_to(F.randn((g.number_of_edges('develops'), 5)), F.cpu())
    return g


@register_case(['batched', 'homo'])
def batched_graph0():
    g1 = dgl.add_self_loop(dgl.graph(([0, 1, 2], [1, 2, 3])))
    g2 = dgl.add_self_loop(dgl.graph(([1, 1], [2, 0])))
    g3 = dgl.add_self_loop(dgl.graph(([0], [1])))
    return dgl.batch([g1, g2, g3])

@register_case(['block', 'bipartite', 'block-biparitite'])
def block_graph0():
    g = dgl.graph(([2, 3, 4], [5, 6, 7]), num_nodes=100)
    g = g.to(F.cpu())
    return dgl.to_block(g)

@register_case(['block'])
def block_graph1():
    g = dgl.heterograph({
            ('user', 'plays', 'game') : ([0, 1, 2], [1, 1, 0]),
            ('user', 'likes', 'game') : ([1, 2, 3], [0, 0, 2]),
            ('store', 'sells', 'game') : ([0, 1, 1], [0, 1, 2]),
        }, device=F.cpu())
    return dgl.to_block(g)

@register_case(['clique'])
def clique():
    g = dgl.graph(([0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]))
    return g

def random_dglgraph(size):
    return dgl.DGLGraph(nx.erdos_renyi_graph(size, 0.3))

def random_graph(size):
    return dgl.from_networkx(nx.erdos_renyi_graph(size, 0.3))

def random_bipartite(size_src, size_dst):
    return dgl.bipartite_from_scipy(ssp.random(size_src, size_dst, 0.1),
                                    utype='_U', etype='_E', vtype='V', )

def random_block(size):
    g = dgl.from_networkx(nx.erdos_renyi_graph(size, 0.1))
    return dgl.to_block(g, np.unique(F.zerocopy_to_numpy(g.edges()[1])))

@register_case(['two_hetero_batch'])
def two_hetero_batch():
    g1 = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ('user', 'follows', 'developer'): ([0, 1], [1, 2]),
        ('user', 'plays', 'game'): ([0, 1, 2, 3], [0, 0, 1, 1])
    })
    g2 = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ('user', 'follows', 'developer'): ([0, 1], [1, 2]),
        ('user', 'plays', 'game'): ([0, 1, 2], [0, 0, 1])
    })
    return [g1, g2]

@register_case(['two_hetero_batch'])
def two_hetero_batch_with_isolated_ntypes():
    g1 = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ('user', 'follows', 'developer'): ([0, 1], [1, 2]),
        ('user', 'plays', 'game'): ([0, 1, 2, 3], [0, 0, 1, 1])
        }, num_nodes_dict={'user': 4, 'game': 2, 'developer': 3, 'platform': 2})
    g2 = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ('user', 'follows', 'developer'): ([0, 1], [1, 2]),
        ('user', 'plays', 'game'): ([0, 1, 2], [0, 0, 1])
        }, num_nodes_dict={'user': 3, 'game': 2, 'developer': 3, 'platform': 3})
    return [g1, g2]
