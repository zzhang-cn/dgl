import dgl
import time
import numpy as np
import scipy as sp
import torch

adj = sp.sparse.random(550, 550, 0.05)
G = dgl.DGLGraph(adj)
N = G.number_of_nodes()
M = G.number_of_edges()
print(N)
print(M)

def udf(edges):
    return {'m' : edges.src['h'] * edges.data['w']}

def time_udf(G):
    dur = []
    for i in range(50):
        if i >= 5:
            t0 = time.time()
        G.update_all(udf, dgl.function.sum('m', 'hh'))
        if i >= 5:
            dur.append(time.time() - t0)
    return np.average(dur)

def time_builtin(G):
    dur = []
    for i in range(50):
        if i >= 5:
            t0 = time.time()
        G.update_all(dgl.function.u_mul_e('h', 'w', 'm'), dgl.function.sum('m', 'hh'))
        if i >= 5:
            dur.append(time.time() - t0)
    return np.average(dur)

G.ndata['h'] = torch.randn((N, 64, 1))
G.edata['w'] = torch.randn((M, 64, 64))
G.to(torch.device('cuda:0'))
print('udf time:', time_udf(G))
print('builtin time:', time_builtin(G))

G.ndata['h'] = torch.randn((N, 64, 64))
G.edata['w'] = torch.randn((M, 64, 64))
G.to(torch.device('cuda:0'))
print('udf time:', time_udf(G))
print('builtin time:', time_builtin(G))
