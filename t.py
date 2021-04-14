from typing import List, Tuple
from dgl.heterograph import DGLHeteroGraph
from torch._C import TupleType
import dgl
from dgl.data import CoraGraphDataset
import dgl.function as fn
import torch
g = CoraGraphDataset()[0]
torch.classes.load_library("/home/ubuntu/dev/torchdgl/build/libTorchDGLGraph.so")
TorchDGLGraph = torch.classes.my_classes.TorchDGLGraph
TorchDGLMetaGraph = torch.classes.my_classes.TorchDGLMetaGraph

src, dst = g.edges()
tg = TorchDGLGraph(src, dst)
from dgl.sparse import _gspmm

def update_all(g: TorchDGLGraph, feat: torch.Tensor):
    h2 = _gspmm(g, 'copy_lhs', "sum", feat, torch.tensor([]))
    return h2

# update_all(g._graph, g.ndata['feat'])
import torch
ss = torch.jit.script(update_all)
tt = g.ndata['feat']
o = update_all(tg, tt)
oo = ss(tg, g.ndata['feat'])