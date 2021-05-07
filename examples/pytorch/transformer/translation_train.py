from modules import *
from loss import *
from optims import *
from dataset import *
from modules.config import *
import numpy as np
import argparse
import torch
from functools import partial
import torch.distributed as dist

def run_epoch(epoch, data_iter, dev_rank, ndev, model, loss_compute, is_train=True):
    universal = isinstance(model, UTransformer)
    with loss_compute:
        for i, g in enumerate(data_iter):
            with T.set_grad_enabled(is_train):
                if universal:
                    output, loss_act = model(g)
                    if is_train: loss_act.backward(retain_graph=True)
                else:
                    output = model(g)
                tgt_y = g.tgt_y
                n_tokens = g.n_tokens
                loss = loss_compute(output, tgt_y, n_tokens)

    if universal:
        for step in range(1, model.MAX_DEPTH + 1):
            print("nodes entering step {}: {:.2f}%".format(step, (1.0 * model.stat[step] / model.stat[0])))
        model.reset_stat()
    print('Epoch {} {}: Dev {} average loss: {}, accuracy {}'.format(
        epoch, "Training" if is_train else "Evaluating",
        dev_rank, loss_compute.avg_loss, loss_compute.accuracy))

def run(dev_id, args):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip=args.master_ip, master_port=args.master_port)
    world_size = args.ngpu
    torch.distributed.init_process_group(backend="nccl",
                                         init_method=dist_init_method,
                                         world_size=world_size,
                                         rank=dev_id)
    gpu_rank = torch.distributed.get_rank()
    assert gpu_rank == dev_id
    main(dev_id, args)

def main(dev_id, args):
    if dev_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(dev_id))
    # Set current device
    th.cuda.set_device(device)
    # Prepare dataset
    dataset = get_dataset(args.dataset)
    V = dataset.vocab_size
    criterion = LabelSmoothing(V, padding_idx=dataset.pad_id, smoothing=0.1)
    dim_model = 512
    # Build graph pool
    graph_pool = GraphPool()
    # Create model
    model = make_model(V, V, N=args.N, dim_model=dim_model,
                       universal=args.universal)
    # Sharing weights between Encoder & Decoder
    model.src_embed.lut.weight = model.tgt_embed.lut.weight
    model.generator.proj.weight = model.tgt_embed.lut.weight
    # Move model to corresponding device
    model, criterion = model.to(device), criterion.to(device)
    # Loss function
    if args.ngpu > 1:
        dev_rank = dev_id # current device id
        ndev = args.ngpu # number of devices (including cpu)
        loss_compute = partial(MultiGPULossCompute, criterion, args.ngpu,
                               args.grad_accum, model)
    else: # cpu or single gpu case
        dev_rank = 0
        ndev = 1
        loss_compute = partial(SimpleLossCompute, criterion, args.grad_accum)

    if ndev > 1:
        for param in model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= ndev

    # Optimizer
    model_opt = NoamOpt(dim_model, 0.1, 4000,
                        T.optim.Adam(model.parameters(), lr=1e-3,
                                     betas=(0.9, 0.98), eps=1e-9))

    # Train & evaluate
    for epoch in range(100):
        start = time.time()
        train_iter = dataset(graph_pool, mode='train', batch_size=args.batch,
                             device=device, dev_rank=dev_rank, ndev=ndev)
        model.train(True)
        run_epoch(epoch, train_iter, dev_rank, ndev, model,
                  loss_compute(opt=model_opt), is_train=True)
        if dev_rank == 0:
            model.att_weight_map = None
            model.eval()
            valid_iter = dataset(graph_pool, mode='valid', batch_size=args.batch,
                                 device=device, dev_rank=dev_rank, ndev=1)
            run_epoch(epoch, valid_iter, dev_rank, 1, model,
                      loss_compute(opt=None), is_train=False)
            end = time.time()
            print("epoch time: {}".format(end - start))

            # Visualize attention
            if args.viz:
                src_seq = dataset.get_seq_by_id(VIZ_IDX, mode='valid', field='src')
                tgt_seq = dataset.get_seq_by_id(VIZ_IDX, mode='valid', field='tgt')[:-1]
                draw_atts(model.att_weight_map, src_seq, tgt_seq, exp_setting, 'epoch_{}'.format(epoch))
            args_filter = ['batch', 'gpus', 'viz', 'master_ip', 'master_port', 'grad_accum', 'ngpu']
            exp_setting = '-'.join('{}'.format(v) for k, v in vars(args).items() if k not in args_filter)
            with open('checkpoints/{}-{}.pkl'.format(exp_setting, epoch), 'wb') as f:
                torch.save(model.state_dict(), f)

if __name__ == '__main__':
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    np.random.seed(1111)
    argparser = argparse.ArgumentParser('training translation model')
    argparser.add_argument('--gpus', default='-1', type=str, help='gpu id')
    argparser.add_argument('--N', default=6, type=int, help='enc/dec layers')
    argparser.add_argument('--dataset', default='multi30k', help='dataset')
    argparser.add_argument('--batch', default=128, type=int, help='batch size')
    argparser.add_argument('--viz', action='store_true',
                           help='visualize attention')
    argparser.add_argument('--universal', action='store_true',
                           help='use universal transformer')
    argparser.add_argument('--master-ip', type=str, default='127.0.0.1',
                           help='master ip address')
    argparser.add_argument('--master-port', type=str, default='12345',
                           help='master port')
    argparser.add_argument('--grad-accum', type=int, default=1,
                           help='accumulate gradients for this many times '
                                'then update weights')
    args = argparser.parse_args()
    print(args)

    devices = list(map(int, args.gpus.split(',')))
    if len(devices) == 1:
        args.ngpu = 0 if devices[0] < 0 else 1
        main(devices[0], args)
    else:
        args.ngpu = len(devices)
        mp = torch.multiprocessing.get_context('spawn')
        procs = []
        for dev_id in devices:
            procs.append(mp.Process(target=run, args=(dev_id, args),
                                    daemon=True))
            procs[-1].start()
        for p in procs:
            p.join()
