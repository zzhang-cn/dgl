import pytest


def test_graph_index_reconstruct():
    from jax.tree_util import tree_flatten, tree_unflatten
    import dgl
    import networkx as nx
    import jax

    g = dgl.heterograph_index.create_unitgraph_from_coo(
        1,
        3,
        3,
        jax.numpy.array([0, 1, 2]),
        jax.numpy.array([0, 1, 2]),
        ["coo"]
    )

    print(type(g))

    def show_example(structured):
        flat, tree = tree_flatten(structured)
        unflattened = tree_unflatten(tree, flat)
        print("structured={}\n  flat={}\n  tree={}\n  unflattened={}".format(
            structured, flat, tree, unflattened))

    show_example(g)

test_graph_index_reconstruct()
