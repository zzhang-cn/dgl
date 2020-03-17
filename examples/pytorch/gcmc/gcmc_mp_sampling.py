"""Training script"""
import os, time
import argparse
import logging
import random
import string
import traceback
import numpy as np
import torch as th
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.multiprocessing import Queue
from torch.nn.parallel import DistributedDataParallel
from _thread import start_new_thread
from functools import wraps
from data import MovieLens
from model import SampleGCMCLayer, GCMCLayer, SampleBiDecoder, BiDecoder
from utils import get_activation, get_optimizer, torch_total_param_num, torch_net_info, MetricLogger
import dgl

from gcmc_sampling import GCMCSampler, Net, evaluate

# According to https://github.com/pytorch/pytorch/issues/17199, this decorator
# is necessary to make fork() and openmp work together.
def thread_wrapped_func(func):
    """
    Wraps a process entry point to make it work with OpenMP.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function

def prepare_mp(g):
    """
    Explicitly materialize the CSR, CSC and COO representation of the given graph
    so that they could be shared via copy-on-write to sampler workers and GPU
    trainers.
    This is a workaround before full shared memory support on heterogeneous graphs.
    """
    for etype in g.canonical_etypes:
        g.in_degree(0, etype=etype)
        g.out_degree(0, etype=etype)
        g.find_edges([0], etype=etype)

def config():
    parser = argparse.ArgumentParser(description='GCMC')
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_dir', type=str, help='The saving directory')
    parser.add_argument('--save_id', type=int, help='The saving log id')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--data_name', default='ml-1m', type=str,
                        help='The dataset name: ml-100k, ml-1m, ml-10m')
    parser.add_argument('--data_test_ratio', type=float, default=0.1) ## for ml-100k the test ration is 0.2
    parser.add_argument('--data_valid_ratio', type=float, default=0.1)
    parser.add_argument('--model_activation', type=str, default="leaky")
    parser.add_argument('--gcn_dropout', type=float, default=0.7)
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    parser.add_argument('--gcn_agg_units', type=int, default=500)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")
    parser.add_argument('--gcn_out_units', type=int, default=75)
    parser.add_argument('--gen_r_num_basis_func', type=int, default=2)
    parser.add_argument('--train_max_epoch', type=int, default=100)
    parser.add_argument('--train_log_interval', type=int, default=1)
    parser.add_argument('--train_valid_interval', type=int, default=1)
    parser.add_argument('--train_optimizer', type=str, default="adam")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_min_lr', type=float, default=0.0001)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--train_decay_patience', type=int, default=5)
    parser.add_argument('--share_param', default=False, action='store_true')
    parser.add_argument('--minibatch_size', type=int, default=10000)
    parser.add_argument('--num_workers_per_gpu', type=int, default=8)

    args = parser.parse_args()
    ### configure save_fir to save all the info
    if args.save_dir is None:
        args.save_dir = args.data_name+"_" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=2))
    if args.save_id is None:
        args.save_id = np.random.randint(20)
    args.save_dir = os.path.join("log", args.save_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args

@thread_wrapped_func
def run(proc_id, n_gpus, args, devices, dataset):
    dev_id = devices[proc_id]
    train_labels = dataset.train_labels
    train_truths = dataset.train_truths
    num_edges = train_truths.shape[0]
    sampler = GCMCSampler(num_edges,
                          args.minibatch_size,
                          dataset)

    seeds = th.arange(num_edges)
    dataloader = DataLoader(
        dataset=seeds,
        batch_size=args.minibatch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers_per_gpu)

    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=dev_id)
    th.cuda.set_device(dev_id)

    nd_possible_rating_values = \
        th.FloatTensor(dataset.possible_rating_values)
    nd_possible_rating_values = nd_possible_rating_values.to(dev_id)

    net = Net(args=args, dev_id=dev_id)
    net = net.to(dev_id)
    if n_gpus > 1:
        net = DistributedDataParallel(net, device_ids=[dev_id], output_device=dev_id)
    rating_loss_net = nn.CrossEntropyLoss()
    learning_rate = args.train_lr
    optimizer = get_optimizer(args.train_optimizer)(net.parameters(), lr=learning_rate)
    print("Loading network finished ...\n")

    ### declare the loss information
    best_valid_rmse = np.inf
    no_better_valid = 0
    best_iter = -1
    count_rmse = 0
    count_num = 0
    count_loss = 0
    dataset.valid_enc_graph.nodes['user'].data['ids'] = \
        th.zeros((dataset.valid_enc_graph.number_of_nodes('user'),), dtype=th.int64)
    dataset.valid_enc_graph.nodes['movie'].data['ids'] = \
        th.zeros((dataset.valid_enc_graph.number_of_nodes('movie'),), dtype=th.int64)
    dataset.test_enc_graph.nodes['user'].data['ids'] = \
        th.zeros((dataset.test_enc_graph.number_of_nodes('user'),), dtype=th.int64)
    dataset.test_enc_graph.nodes['movie'].data['ids'] = \
        th.zeros((dataset.test_enc_graph.number_of_nodes('movie'),), dtype=th.int64)
    print("Start training ...")
    dur = []
    iter_idx = 1

    for epoch in range(1, args.train_max_epoch):
        if epoch > 1:
            t0 = time.time()
        net.train()
        for step, sample_data in enumerate(dataloader):           
            head_frontier, tail_frontier, head_feat, tail_feat, \
            head_id, tail_id, true_relation_labels, true_relation_ratings = sample_data
            head_feat = head_feat.to(dev_id)
            tail_feat = tail_feat.to(dev_id)

            pred_ratings = net(head_frontier, tail_frontier, head_feat, tail_feat, head_id, tail_id)
            loss = rating_loss_net(pred_ratings, true_relation_labels.to(dev_id)).mean()
            count_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), args.train_grad_clip)
            optimizer.step()

            if proc_id == 0 and iter_idx == 1:
                print("Total #Param of net: %d" % (torch_total_param_num(net)))

            real_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                                nd_possible_rating_values.view(1, -1)).sum(dim=1)
            rmse = ((real_pred_ratings - true_relation_ratings.to(dev_id)) ** 2).sum()
            count_rmse += rmse.item()
            count_num += pred_ratings.shape[0]

            if iter_idx % args.train_log_interval == 0:
                logging_str = "Iter={}, loss={:.4f}, rmse={:.4f}".format(
                    iter_idx, count_loss/iter_idx, count_rmse/count_num)
                count_rmse = 0
                count_num = 0

            if iter_idx % args.train_log_interval == 0:
                print("[{}] {}".format(proc_id, logging_str))

            iter_idx += 1
        if epoch > 1:
            epoch_time = time.time() - t0
            print("Epoch {} time {}".format(epoch, epoch_time))

        if epoch % args.train_valid_interval == 0:
            if n_gpus > 1:
                th.distributed.barrier()
            if proc_id == 0:
                valid_rmse = evaluate(args=args, dev_id=dev_id, net=net, dataset=dataset, segment='valid')
                logging_str += ',\tVal RMSE={:.4f}'.format(valid_rmse)

                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    no_better_valid = 0
                    best_iter = iter_idx
                    test_rmse = evaluate(args=args, dev_id=dev_id, net=net, dataset=dataset, segment='test')
                    best_test_rmse = test_rmse
                    logging_str += ', Test RMSE={:.4f}'.format(test_rmse)
                else:
                    no_better_valid += 1
                    if no_better_valid > args.train_decay_patience:
                        new_lr = max(learning_rate * args.train_lr_decay_factor, args.train_min_lr)
                        if new_lr < learning_rate:
                            logging.info("\tChange the LR to %g" % new_lr)
                            learning_rate = new_lr
                            for p in optimizer.param_groups:
                                p['lr'] = learning_rate
                            no_better_valid = 0
                            print("Change the LR to %g" % new_lr)
            # sync on evalution
            if n_gpus > 1:
                th.distributed.barrier()

        print(logging_str)

if __name__ == '__main__':
    args = config()
    args.mix_cpu_gpu = True
    args.use_one_hot_fea = False

    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)

    # For GCMC based on sampling, we require node has its own features.
    # Otherwise (node_id is the feature), the model can not scale
    dataset = MovieLens(args.data_name,
                        'cpu',
                        use_one_hot_fea=args.use_one_hot_fea,
                        symm=args.gcn_agg_norm_symm,
                        test_ratio=args.data_test_ratio,
                        valid_ratio=args.data_valid_ratio)
    print("Loading data finished ...\n")

    args.src_in_units = dataset.user_feature_shape[1]
    args.dst_in_units = dataset.movie_feature_shape[1]
    args.rating_vals = dataset.possible_rating_values

    if n_gpus == 1:
        run(0, n_gpus, args, devices, dataset)
    else:
        prepare_mp(dataset.train_enc_graph)
        prepare_mp(dataset.train_dec_graph)
        procs = []
        for proc_id in range(n_gpus):
            p = mp.Process(target=run, args=(proc_id, n_gpus, args, devices, dataset))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
