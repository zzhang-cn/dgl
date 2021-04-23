"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn
Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""
import argparse, gc
import numpy as np
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import dgl
from dgl import DGLGraph
from functools import partial

from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from model import RelGraphEmbedLayer
from dgl.nn import RelGraphConv
from utils import thread_wrapped_func
import tqdm

from ogb.nodeproppred import DglNodePropPredDataset

class EntityClassify(nn.Module):
    """ Entity classification class for RGCN
    Parameters
    ----------
    device : int
        Device to run the layer.
    num_nodes : int
        Number of nodes.
    h_dim : int
        Hidden dim size.
    out_dim : int
        Output dim size.
    num_rels : int
        Numer of relation types.
    num_bases : int, optional
        Number of bases. If is none, use number of relations.
        Default None
    num_hidden_layers : int, optional
        Number of hidden RelGraphConv Layer
        Default 1
    dropout : float, optional
        Dropout.
        Default 0
    use_self_loop : bool, optional
        Use self loop if True.
        Default True
    low_mem : bool, optional
        True to use low memory implementation of relation message passing function
        trade speed with memory consumption
        Default True
    layer_norm : bool, optional
        True to use layer norm.
        Default False
    """
    def __init__(self,
                 device,
                 num_nodes,
                 h_dim,
                 out_dim,
                 num_rels,
                 num_bases=None,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False,
                 low_mem=True,
                 layer_norm=False):
        super(EntityClassify, self).__init__()
        self.device = th.device(device if device >= 0 else 'cpu')
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.low_mem = low_mem
        self.layer_norm = layer_norm

        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConv(
            self.h_dim, self.h_dim, self.num_rels, "basis",
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            low_mem=self.low_mem, dropout=self.dropout, layer_norm = layer_norm))
        # h2h
        for idx in range(self.num_hidden_layers):
            self.layers.append(RelGraphConv(
                self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                low_mem=self.low_mem, dropout=self.dropout, layer_norm = layer_norm))
        # h2o
        self.layers.append(RelGraphConv(
            self.h_dim, self.out_dim, self.num_rels, "basis",
            self.num_bases, activation=None,
            self_loop=self.use_self_loop,
            low_mem=self.low_mem, layer_norm = layer_norm))

    def forward(self, blocks, feats, norm=None):
        if blocks is None:
            # full graph training
            blocks = [self.g] * len(self.layers)
        h = feats
        for layer, block in zip(self.layers, blocks):
            block = block.to(self.device)
            h = layer(block, h, block.edata['etype'], block.edata['norm'])
        return h

def gen_norm(g):
    _, v, eid = g.all_edges(form='all')
    _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
    degrees = count[inverse_index]
    norm = th.ones(eid.shape[0], device=eid.device) / degrees
    norm = norm.unsqueeze(1)
    g.edata['norm'] = norm

class NeighborSampler:
    """Neighbor sampler
    Parameters
    ----------
    g : DGLHeterograph
        Full graph
    target_idx : tensor
        The target training node IDs in g
    fanouts : list of int
        Fanout of each hop starting from the seed nodes. If a fanout is None,
        sample full neighbors.
    """
    def __init__(self, g, target_idx, fanouts):
        self.g = g
        self.target_idx = target_idx
        self.fanouts = fanouts

    """Do neighbor sample
    Parameters
    ----------
    seeds :
        Seed nodes
    Returns
    -------
    tensor
        Seed nodes, also known as target nodes
    blocks
        Sampled subgraphs
    """
    def sample_blocks(self, seeds):
        blocks = []
        etypes = []
        norms = []
        ntypes = []
        seeds = th.tensor(seeds).long()
        cur = self.target_idx[seeds]
        for fanout in self.fanouts:
            if fanout is None or fanout == -1:
                frontier = dgl.in_subgraph(self.g, cur)
            else:
                frontier = dgl.sampling.sample_neighbors(self.g, cur, fanout)
            block = dgl.to_block(frontier, cur)
            gen_norm(block)
            cur = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return seeds, blocks

def evaluate(model, embed_layer, eval_loader, node_feats):
    model.eval()
    embed_layer.eval()
    eval_logits = []
    eval_seeds = []

    with th.no_grad():
        th.cuda.empty_cache()
        for sample_data in tqdm.tqdm(eval_loader):
            seeds, blocks = sample_data

            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata['ntype'],
                                blocks[0].srcdata['type_id'],
                                node_feats)
            logits = model(blocks, feats)
            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(seeds.cpu().detach())

    eval_logits = th.cat(eval_logits)
    eval_seeds = th.cat(eval_seeds)

    return eval_logits, eval_seeds

@thread_wrapped_func
def run(proc_id, n_gpus, n_cpus, args, devices, dataset, split, queue=None):
    dev_id = devices[proc_id] if devices[proc_id] != 'cpu' else -1
    g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, \
        train_idx, val_idx, test_idx, labels = dataset
    if split is not None:
        train_seed, val_seed, test_seed = split
        train_idx = train_idx[train_seed]
        val_idx = val_idx[val_seed]
        test_idx = test_idx[test_seed]

    fanouts = [int(fanout) for fanout in args.fanout.split(',')]
    node_tids = g.ndata[dgl.NTYPE]
    sampler = NeighborSampler(g, target_idx, fanouts)
    loader = DataLoader(dataset=train_idx.numpy(),
                        batch_size=args.batch_size,
                        collate_fn=sampler.sample_blocks,
                        shuffle=True,
                        num_workers=args.num_workers)

    # validation sampler
    val_sampler = NeighborSampler(g, target_idx, fanouts)
    val_loader = DataLoader(dataset=val_idx.numpy(),
                            batch_size=args.batch_size,
                            collate_fn=val_sampler.sample_blocks,
                            shuffle=False,
                            num_workers=args.num_workers)

    # test sampler
    test_sampler = NeighborSampler(g, target_idx, [None] * args.n_layers)
    test_loader = DataLoader(dataset=test_idx.numpy(),
                             batch_size=args.eval_batch_size,
                             collate_fn=test_sampler.sample_blocks,
                             shuffle=False,
                             num_workers=args.num_workers)

    world_size = n_gpus
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        backend = 'nccl'

        # using sparse embedding or usig mix_cpu_gpu model (embedding model can not be stored in GPU)
        if dev_id < 0 or args.dgl_sparse is False:
            backend = 'gloo'
        print("backend using {}".format(backend))
        th.distributed.init_process_group(backend=backend,
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)

    # node features
    # None for one-hot feature, if not none, it should be the feature tensor.
    #
    embed_layer = RelGraphEmbedLayer(dev_id if args.embedding_gpu else -1,
                                     dev_id,
                                     g.number_of_nodes(),
                                     node_tids,
                                     num_of_ntype,
                                     node_feats,
                                     args.n_hidden,
                                     dgl_sparse=args.dgl_sparse)

    # create model
    # all model params are in device.
    model = EntityClassify(dev_id,
                           g.number_of_nodes(),
                           args.n_hidden,
                           num_classes,
                           num_rels,
                           num_bases=args.n_bases,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                           use_self_loop=args.use_self_loop,
                           low_mem=args.low_mem,
                           layer_norm=args.layer_norm)

    if dev_id >= 0 and n_gpus == 1:
        th.cuda.set_device(dev_id)
        labels = labels.to(dev_id)
        model.cuda(dev_id)
        # with dgl_sparse emb, only node embedding is not in GPU
        if args.dgl_sparse:
            embed_layer.cuda(dev_id)

    if n_gpus > 1:
        labels = labels.to(dev_id)
        if dev_id >= 0:
            model.cuda(dev_id)
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
        if args.dgl_sparse:
            embed_layer.cuda(dev_id)
            if len(list(embed_layer.parameters())) > 0:
                embed_layer = DistributedDataParallel(embed_layer, device_ids=[dev_id], output_device=dev_id)
        else:
            if len(list(embed_layer.parameters())) > 0:
                embed_layer = DistributedDataParallel(embed_layer, device_ids=None, output_device=None)

    # optimizer
    dense_params = list(model.parameters())
    if args.node_feats:
        if  n_gpus > 1:
            dense_params += list(embed_layer.module.embeds.parameters())
        else:
            dense_params += list(embed_layer.embeds.parameters())
    optimizer = th.optim.Adam(dense_params, lr=args.lr, weight_decay=args.l2norm)

    if args.dgl_sparse:
        all_params = list(model.parameters()) + list(embed_layer.parameters())
        optimizer = th.optim.Adam(all_params, lr=args.lr, weight_decay=args.l2norm)
        if n_gpus > 1 and isinstance(embed_layer, DistributedDataParallel):
            dgl_emb = embed_layer.module.dgl_emb
        else:
            dgl_emb = embed_layer.dgl_emb
        emb_optimizer = dgl.optim.SparseAdam(params=dgl_emb, lr=args.sparse_lr, eps=1e-8) if len(dgl_emb) > 0 else None
    else:
        if n_gpus > 1:
            embs = list(embed_layer.module.node_embeds.parameters())
        else:
            embs = list(embed_layer.node_embeds.parameters())
        emb_optimizer = th.optim.SparseAdam(embs, lr=args.sparse_lr) if len(embs) > 0 else None

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []

    train_time = 0
    validation_time = 0
    test_time = 0
    last_val_acc = 0.0
    do_test = False
    if n_gpus > 1 and n_cpus - args.num_workers > 0:
        th.set_num_threads(n_cpus-args.num_workers)
    for epoch in range(args.n_epochs):
        tstart = time.time()
        model.train()
        embed_layer.train()

        for i, sample_data in enumerate(loader):
            seeds, blocks = sample_data
            t0 = time.time()
            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata['ntype'],
                                blocks[0].srcdata['type_id'],
                                node_feats)
            logits = model(blocks, feats)
            loss = F.cross_entropy(logits, labels[seeds])
            t1 = time.time()
            optimizer.zero_grad()
            if emb_optimizer is not None:
                emb_optimizer.zero_grad()

            loss.backward()
            if emb_optimizer is not None:
                emb_optimizer.step()
            optimizer.step()
            t2 = time.time()

            forward_time.append(t1 - t0)
            backward_time.append(t2 - t1)
            train_acc = th.sum(logits.argmax(dim=1) == labels[seeds]).item() / len(seeds)
            if i % 100 == 0 and proc_id == 0:
                print("Train Accuracy: {:.4f} | Train Loss: {:.4f}".
                    format(train_acc, loss.item()))
        gc.collect()
        print("Epoch {:05d}:{:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
            format(epoch, args.n_epochs, forward_time[-1], backward_time[-1]))
        tend = time.time()
        train_time += (tend - tstart)

        def collect_eval():
            eval_logits = []
            eval_seeds = []
            for i in range(n_gpus):
                log = queue.get()
                eval_l, eval_s = log
                eval_logits.append(eval_l)
                eval_seeds.append(eval_s)
            eval_logits = th.cat(eval_logits)
            eval_seeds = th.cat(eval_seeds)
            eval_loss = F.cross_entropy(eval_logits, labels[eval_seeds].cpu()).item()
            eval_acc = th.sum(eval_logits.argmax(dim=1) == labels[eval_seeds].cpu()).item() / len(eval_seeds)

            return eval_loss, eval_acc

        vstart = time.time()
        if (queue is not None) or (proc_id == 0):
            val_logits, val_seeds = evaluate(model, embed_layer, val_loader, node_feats)
            if queue is not None:
                queue.put((val_logits, val_seeds))

            # gather evaluation result from multiple processes
            if proc_id == 0:
                val_loss, val_acc = collect_eval() if queue is not None else \
                    (F.cross_entropy(val_logits, labels[val_seeds].cpu()).item(), \
                    th.sum(val_logits.argmax(dim=1) == labels[val_seeds].cpu()).item() / len(val_seeds))

                do_test = val_acc > last_val_acc
                last_val_acc = val_acc
                print("Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
                        format(val_acc, val_loss))
        if n_gpus > 1:
            th.distributed.barrier()
            if proc_id == 0:
                for i in range(1, n_gpus):
                    queue.put(do_test)
            else:
                do_test = queue.get()

        vend = time.time()
        validation_time += (vend - vstart)

        if epoch > 0 and do_test:
            tstart = time.time()
            if (queue is not None) or (proc_id == 0):
                test_logits, test_seeds = evaluate(model, embed_layer, test_loader, node_feats)
                if queue is not None:
                    queue.put((test_logits, test_seeds))

                # gather evaluation result from multiple processes
                if proc_id == 0:
                    test_loss, test_acc = collect_eval() if queue is not None else \
                        (F.cross_entropy(test_logits, labels[test_seeds].cpu()).item(), \
                        th.sum(test_logits.argmax(dim=1) == labels[test_seeds].cpu()).item() / len(test_seeds))
                    print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss))
                    print()
            tend = time.time()
            test_time += (tend-tstart)

            # sync for test
            if n_gpus > 1:
                th.distributed.barrier()

    print("{}/{} Mean forward time: {:4f}".format(proc_id, n_gpus,
                                                  np.mean(forward_time[len(forward_time) // 4:])))
    print("{}/{} Mean backward time: {:4f}".format(proc_id, n_gpus,
                                                   np.mean(backward_time[len(backward_time) // 4:])))
    if proc_id == 0:
        print("Final Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss))
        print("Train {}s, valid {}s, test {}s".format(train_time, validation_time, test_time))

def main(args, devices):
    # load graph data
    ogb_dataset = False
    if args.dataset == 'aifb':
        dataset = AIFBDataset()
    elif args.dataset == 'mutag':
        dataset = MUTAGDataset()
    elif args.dataset == 'bgs':
        dataset = BGSDataset()
    elif args.dataset == 'am':
        dataset = AMDataset()
    elif args.dataset == 'ogbn-mag':
        dataset = DglNodePropPredDataset(name=args.dataset)
        ogb_dataset = True
    else:
        raise ValueError()

    if ogb_dataset is True:
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]['paper']
        val_idx = split_idx["valid"]['paper']
        test_idx = split_idx["test"]['paper']
        hg_orig, labels = dataset[0]
        subgs = {}
        for etype in hg_orig.canonical_etypes:
            u, v = hg_orig.all_edges(etype=etype)
            subgs[etype] = (u, v)
            subgs[(etype[2], 'rev-'+etype[1], etype[0])] = (v, u)
        hg = dgl.heterograph(subgs)
        hg.nodes['paper'].data['feat'] = hg_orig.nodes['paper'].data['feat']
        labels = labels['paper'].squeeze()

        num_rels = len(hg.canonical_etypes)
        num_of_ntype = len(hg.ntypes)
        num_classes = dataset.num_classes
        if args.dataset == 'ogbn-mag':
            category = 'paper'
        print('Number of relations: {}'.format(num_rels))
        print('Number of class: {}'.format(num_classes))
        print('Number of train: {}'.format(len(train_idx)))
        print('Number of valid: {}'.format(len(val_idx)))
        print('Number of test: {}'.format(len(test_idx)))

    else:
        # Load from hetero-graph
        hg = dataset[0]

        num_rels = len(hg.canonical_etypes)
        num_of_ntype = len(hg.ntypes)
        category = dataset.predict_category
        num_classes = dataset.num_classes
        train_mask = hg.nodes[category].data.pop('train_mask')
        test_mask = hg.nodes[category].data.pop('test_mask')
        labels = hg.nodes[category].data.pop('labels')
        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

        # AIFB, MUTAG, BGS and AM datasets do not provide validation set split.
        # Split train set into train and validation if args.validation is set
        # otherwise use train set as the validation set.
        if args.validation:
            val_idx = train_idx[:len(train_idx) // 5]
            train_idx = train_idx[len(train_idx) // 5:]
        else:
            val_idx = train_idx

    node_feats = []
    for ntype in hg.ntypes:
        if len(hg.nodes[ntype].data) == 0 or args.node_feats is False:
            node_feats.append(hg.number_of_nodes(ntype))
        else:
            assert len(hg.nodes[ntype].data) == 1
            feat = hg.nodes[ntype].data.pop('feat')
            node_feats.append(feat.share_memory_())

    # get target category id
    category_id = len(hg.ntypes)
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i
        print('{}:{}'.format(i, ntype))

    g = dgl.to_homogeneous(hg)
    g.ndata['ntype'] = g.ndata[dgl.NTYPE]
    g.ndata['ntype'].share_memory_()
    g.edata['etype'] = g.edata[dgl.ETYPE]
    g.edata['etype'].share_memory_()
    g.ndata['type_id'] = g.ndata[dgl.NID]
    g.ndata['type_id'].share_memory_()
    node_ids = th.arange(g.number_of_nodes())

    # find out the target node ids
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]
    target_idx.share_memory_()
    train_idx.share_memory_()
    val_idx.share_memory_()
    test_idx.share_memory_()
    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    g.create_formats_()

    n_gpus = len(devices)
    n_cpus = mp.cpu_count()
    # cpu
    if devices[0] == -1 and n_gpus == 1:
        run(0, 0, n_cpus, args, ['cpu'],
            (g, node_feats, num_of_ntype, num_classes, num_rels, target_idx,
             train_idx, val_idx, test_idx, labels), None, None)
    # gpu
    elif n_gpus == 1:
        run(0, n_gpus, n_cpus, args, devices,
            (g, node_feats, num_of_ntype, num_classes, num_rels, target_idx,
            train_idx, val_idx, test_idx, labels), None, None)
    # multi gpu
    else:
        queue = mp.Queue(n_gpus)
        procs = []
        num_train_seeds = train_idx.shape[0]
        num_valid_seeds = val_idx.shape[0]
        num_test_seeds = test_idx.shape[0]
        train_seeds = th.randperm(num_train_seeds)
        valid_seeds = th.randperm(num_valid_seeds)
        test_seeds = th.randperm(num_test_seeds)
        tseeds_per_proc = num_train_seeds // n_gpus
        vseeds_per_proc = num_valid_seeds // n_gpus
        tstseeds_per_proc = num_test_seeds // n_gpus
        for proc_id in range(n_gpus):
            # we have multi-gpu for training, evaluation and testing
            # so split trian set, valid set and test set into num-of-gpu parts.
            proc_train_seeds = train_seeds[proc_id * tseeds_per_proc :
                                           (proc_id + 1) * tseeds_per_proc \
                                           if (proc_id + 1) * tseeds_per_proc < num_train_seeds \
                                           else num_train_seeds]
            proc_valid_seeds = valid_seeds[proc_id * vseeds_per_proc :
                                           (proc_id + 1) * vseeds_per_proc \
                                           if (proc_id + 1) * vseeds_per_proc < num_valid_seeds \
                                           else num_valid_seeds]
            proc_test_seeds = test_seeds[proc_id * tstseeds_per_proc :
                                         (proc_id + 1) * tstseeds_per_proc \
                                         if (proc_id + 1) * tstseeds_per_proc < num_test_seeds \
                                         else num_test_seeds]
            p = mp.Process(target=run, args=(proc_id, n_gpus, n_cpus // n_gpus, args, devices,
                                             (g, node_feats, num_of_ntype, num_classes, num_rels, target_idx,
                                             train_idx, val_idx, test_idx, labels),
                                             (proc_train_seeds, proc_valid_seeds, proc_test_seeds),
                                             queue))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()


def config():
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--gpu", type=str, default='0',
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--sparse-lr", type=float, default=2e-2,
            help="sparse embedding learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--fanout", type=str, default="4, 4",
            help="Fan-out of neighbor sampling.")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.add_argument("--batch-size", type=int, default=100,
            help="Mini-batch size. ")
    parser.add_argument("--eval-batch-size", type=int, default=32,
            help="Mini-batch size. ")
    parser.add_argument("--num-workers", type=int, default=0,
            help="Number of workers for dataloader.")
    parser.add_argument("--low-mem", default=False, action='store_true',
            help="Whether use low mem RelGraphCov")
    parser.add_argument("--dgl-sparse", default=False, action='store_true',
            help='Use sparse embedding for node embeddings.')
    parser.add_argument("--embedding-gpu", default=False, action='store_true',
            help='Store the node embeddings on the GPU.')
    parser.add_argument('--node-feats', default=False, action='store_true',
            help='Whether use node features')
    parser.add_argument('--layer-norm', default=False, action='store_true',
            help='Use layer norm')
    parser.set_defaults(validation=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = config()
    devices = list(map(int, args.gpu.split(',')))
    print(args)
    main(args, devices)
