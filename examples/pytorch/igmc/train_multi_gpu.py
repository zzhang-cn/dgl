"""Training IGMC model on the MovieLens dataset."""

import os
import sys
import time
import glob
import random
import argparse
from shutil import copy

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import dgl

import traceback
from functools import wraps
from _thread import start_new_thread
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from model import IGMC
from data import MovieLens
from dataset import MovieLensDataset, collate_movielens 
from utils import MetricLogger

os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()

# According to https://github.com/pytorch/pytorch/issues/17199, this decorator
# is necessary to make fork() and openmp work together.
def thread_wrapped_func(func):
    """
    Wraps a process entry point to make it work with OpenMP.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = mp.Queue()
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

def evaluate(model, loader, device):
    # Evaluate RMSE
    model.eval()
    device = th.device(device)

    mse = 0.
    for batch in loader:
        with th.no_grad():
            preds = model(batch[0].to(device))
        labels = batch[1].to(device)
        mse += ((preds - labels) ** 2).sum().item()
    mse /= len(loader.dataset)
    return np.sqrt(mse)

def adj_rating_reg(model):
    if isinstance(model, DistributedDataParallel):
        model = model.module

    arr_loss = 0
    for conv in model.convs:
        weight = conv.weight.view(conv.num_bases, conv.in_feat * conv.out_feat)
        weight = th.matmul(conv.w_comp, weight).view(conv.num_rels, conv.in_feat, conv.out_feat)
        arr_loss += th.sum((weight[1:, :, :] - weight[:-1, :, :])**2)
    return arr_loss

def train_epoch(proc_id, n_gpus, model, loss_fn, optimizer, arr_lambda, loader, device, log_interval):
    model.train()
    device = th.device(device)

    epoch_loss = 0.
    iter_loss = 0.
    iter_mse = 0.
    iter_cnt = 0
    iter_dur = []
    for iter_idx, batch in enumerate(loader, start=1):
        t_start = time.time()

        preds = model(batch[0].to(device))
        labels = batch[1].to(device)
        loss = loss_fn(preds, labels).mean() + arr_lambda * adj_rating_reg(model)
        
        optimizer.zero_grad()
        loss.backward()
        if n_gpus > 1:
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    th.distributed.all_reduce(param.grad.data,
                                                op=th.distributed.ReduceOp.SUM)
                    param.grad.data /= n_gpus
        optimizer.step()

        if proc_id == 0:
            epoch_loss += loss.item() * preds.shape[0]
            iter_loss += loss.item() * preds.shape[0]
            iter_mse += ((preds - labels) ** 2).sum().item()
            iter_cnt += preds.shape[0]
            iter_dur.append(time.time() - t_start)

            if iter_idx % log_interval == 0:
                print("Iter={}, loss={:.4f}, mse={:.4f}, time={:.4f}".format(
                    iter_idx, iter_loss/iter_cnt, iter_mse/iter_cnt, np.average(iter_dur)))
                iter_loss = 0.
                iter_mse = 0.
                iter_cnt = 0
    return epoch_loss / len(loader.dataset)

@thread_wrapped_func
def train(proc_id, n_gpus, args, devices, movielens):
    # Start up distributed training, if enabled.
    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)
    th.cuda.set_device(dev_id)
    # set random seed in each gpu
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)
    dgl.random.seed(args.seed)

    # Split train_dataset and set dataloader
    train_rating_pairs = th.split(th.stack(movielens.train_rating_pairs), 
                                len(movielens.train_rating_values)//args.n_gpus, 
                                dim=1)[proc_id]
    train_rating_values = th.split(movielens.train_rating_values, 
                                len(movielens.train_rating_values)//args.n_gpus, 
                                dim=0)[proc_id]

    train_dataset = MovieLensDataset(
        train_rating_pairs, train_rating_values, movielens.train_graph, 
        args.hop, args.sample_ratio, args.max_nodes_per_hop)
    train_loader = th.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers, collate_fn=collate_movielens)
    if proc_id == 0:
        if args.testing:
            test_dataset = MovieLensDataset(
                movielens.test_rating_pairs, movielens.test_rating_values, movielens.train_graph, 
                args.hop, args.sample_ratio, args.max_nodes_per_hop)
        else:
            test_dataset = MovieLensDataset(
                movielens.valid_rating_pairs, movielens.valid_rating_pairs, movielens.train_graph, 
                args.hop, args.sample_ratio, args.max_nodes_per_hop)
        test_loader = th.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.num_workers, collate_fn=collate_movielens)

    in_feats = (args.hop+1)*2 # + movielens.train_graph.ndata['refex'].shape[1]
    model = IGMC(in_feats=in_feats, 
                 latent_dim=[32, 32, 32, 32],
                 num_relations=5, #dataset_base.num_rating, 
                 num_bases=4, 
                 regression=True, 
                 edge_dropout=args.edge_dropout,
                #  side_features=args.use_features,
                #  n_side_features=n_features,
                #  multiply_by=args.multiply_by
            ).to(dev_id)
    if n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
    loss_fn = nn.MSELoss().to(dev_id)
    optimizer = optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=0)

    if proc_id == 0:
        print("Loading network finished ...\n")
        # prepare the logger
        logger = MetricLogger(args.save_dir, args.valid_log_interval)

        best_epoch = 0
        best_rmse = np.inf
        print("Start training ...")

    for epoch_idx in range(1, args.train_epochs+1):
        if proc_id == 0:
            print ('Epoch', epoch_idx)
    
        train_loss = train_epoch(proc_id, n_gpus, 
                                model, loss_fn, optimizer, args.arr_lambda, 
                                train_loader, dev_id, args.train_log_interval)

        if n_gpus > 1:
            th.distributed.barrier()
        if proc_id == 0:
            test_rmse = evaluate(model, test_loader, dev_id)
            eval_info = {
                'epoch': epoch_idx,
                'train_loss': train_loss,
                'test_rmse': test_rmse,
            }
            print('=== Epoch {}, train loss {:.6f}, test rmse {:.6f} ==='.format(*eval_info.values()))

            if epoch_idx % args.train_lr_decay_step == 0:
                for param in optimizer.param_groups:
                    param['lr'] = args.train_lr_decay_factor * param['lr']

            logger.log(eval_info, model, optimizer)
            if best_rmse > test_rmse:
                best_rmse = test_rmse
                best_epoch = epoch_idx

    if n_gpus > 1:
        th.distributed.barrier()
    if proc_id == 0:
        eval_info = "Training ends. The best testing rmse is {:.6f} at epoch {}".format(best_rmse, best_epoch)
        print(eval_info)
        with open(os.path.join(args.save_dir, 'log.txt'), 'a') as f:
            f.write(eval_info)
            
def config():
    parser = argparse.ArgumentParser(description='IGMC')
    # general settings
    parser.add_argument('--testing', action='store_true', default=False,
                        help='if set, use testing mode which splits all ratings into train/test;\
                        otherwise, use validation model which splits all ratings into \
                        train/val/test and evaluate on val only')
    parser.add_argument('--gpu', default='0', type=str,
                        help="Comma separated list of GPU device IDs.")
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--data_name', default='ml-100k', type=str,
                        help='The dataset name: ml-100k, ml-1m')
    parser.add_argument('--data_test_ratio', type=float, default=0.1) # for ml-100k the test ration is 0.2
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--data_valid_ratio', type=float, default=0.2)
    # parser.add_argument('--ensemble', action='store_true', default=False,
    #                     help='if True, load a series of model checkpoints and ensemble the results')               
    parser.add_argument('--train_log_interval', type=int, default=100)
    parser.add_argument('--valid_log_interval', type=int, default=10)
    parser.add_argument('--save_appendix', type=str, default='debug', 
                        help='what to append to save-names when saving results')
    # subgraph extraction settings
    parser.add_argument('--hop', default=1, metavar='S', 
                        help='enclosing subgraph hop number')
    parser.add_argument('--sample_ratio', type=float, default=1.0, 
                        help='if < 1, subsample nodes per hop according to the ratio')
    parser.add_argument('--max_nodes_per_hop', type=int, default=200, 
                        help='if > 0, upper bound the # nodes per hop by another subsampling')
    # parser.add_argument('--use_features', action='store_true', default=False,
    #                     help='whether to use node features (side information)')
    # edge dropout settings
    parser.add_argument('--edge_dropout', type=float, default=0.2, 
                        help='if not 0, random drops edges from adjacency matrix with this prob')
    parser.add_argument('--force_undirected', action='store_true', default=False, 
                        help='in edge dropout, force (x, y) and (y, x) to be dropped together')
    # optimization settings
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--train_min_lr', type=float, default=1e-6)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--train_lr_decay_step', type=int, default=50)
    parser.add_argument('--train_epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--arr_lambda', type=float, default=0.001)
    parser.add_argument('--num_rgcn_bases', type=int, default=4)
                
    args = parser.parse_args()
    args.devices = list(map(int, args.gpu.split(',')))
    args.n_gpus = len(args.devices)

    ### set save_dir according to localtime and test mode
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    val_test_appendix = 'testmode' if args.testing else 'valmode'
    local_time = time.strftime('%y%m%d%H%M', time.localtime())
    args.save_dir = os.path.join(
        file_dir, 'log/{}_{}_{}_{}'.format(
            args.data_name, args.save_appendix, val_test_appendix, local_time
        )
    )
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir) 
    print(args)

    # backup current .py files
    for f in glob.glob(r"*.py"):
        copy(f, args.save_dir)

    # save command line input
    cmd_input = 'python3 ' + ' '.join(sys.argv)
    with open(os.path.join(args.save_dir, 'cmd_input.txt'), 'a') as f:
        f.write(cmd_input)
        f.write("\n")
    print('Command line input: ' + cmd_input + ' is saved.')
    
    return args

if __name__ == '__main__':
    args = config()
    random.seed(args.seed)
    np.random.seed(args.seed)

    movielens = MovieLens(args.data_name, testing=args.testing,
                            test_ratio=args.data_test_ratio, valid_ratio=args.data_valid_ratio)

    if args.n_gpus == 1:
        train(0, args.n_gpus, args, args.devices, movielens)
    else:
        procs = []
        for proc_id in range(args.n_gpus):
            p = mp.Process(target=train, args=(proc_id, args.n_gpus, args, args.devices, movielens))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
