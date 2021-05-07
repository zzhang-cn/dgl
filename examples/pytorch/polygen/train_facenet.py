from modules import *
from loss import *
from optims import *
from dataset import *
import numpy as np
import argparse
import torch
from functools import partial
import torch.distributed as dist

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
    # Considering data paralellism
    train_dataset = FaceDataset(args.dataset, 'train', device, dev_id, args.ngpu)
    # TODO: debugging nan, set shuffle to false
    train_loader = DataLoader(train_dataset, batch_size=args.batch//args.ngpu, shuffle=True, num_workers=args.workers_per_loader, collate_fn=collate_facegraphs)
    train_iter = iter(train_loader)

    # Config loss
    criterion = torch.nn.CrossEntropyLoss()
    # Create model, set the sota setting as the default params
    model = make_face_model(N=args.N, dim_model=args.dim_model)
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
    # All reduce to make sure initialized model is the same
    if ndev > 1:
        for param in model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= ndev

    # Optimizer
    model_opt = NoamOpt(args.dim_model, 0.1, 4000,
                        T.optim.Adam(model.parameters(), lr=3e-4,
                                     betas=(0.9, 0.98), eps=1e-9))
    train_loss_compute = loss_compute(opt=model_opt)
    # Logging and testing only run on device 0
    if dev_id == 0:
        # Create ckpt dir
        os.makedirs(args.ckpt_dir, exist_ok=True)
        log_path = os.path.join(args.ckpt_dir, 'log.txt')
        print (log_path)
        log_f = open(log_path, 'w')
        test_dataset = FaceDataset(args.dataset, 'test', device)
        test_loader = DataLoader(test_dataset, batch_size=args.batch//args.ngpu, shuffle=False, num_workers=args.workers_per_loader, collate_fn=collate_facegraphs)
        test_loss_compute = partial(SimpleLossCompute, criterion, args.grad_accum)(opt=None)
 
    train_iter_num = 0
    model.train(True)
    while True:
        # train step
        try:
            train_batch = train_iter.next()
        except:
            train_iter = iter(train_loader)
            train_batch = train_iter.next()
        with train_loss_compute:
            with T.set_grad_enabled(True):
                output = model(train_batch)
                tgt_y = train_batch.tgt_y
                n_tokens = train_batch.n_tokens
                train_loss = train_loss_compute(output, tgt_y, n_tokens)
        train_iter_num += 1

        # testing logging
        if train_iter_num % args.log_interval == 0 and dev_rank == 0:
            model.eval()
            # test step
            try:
                test_batch = test_iter.next()
            except:
                test_iter = iter(test_loader)
                test_batch = test_iter.next()
            with test_loss_compute:
                with T.set_grad_enabled(False):
                    output = model(test_batch)
                    tgt_y = test_batch.tgt_y
                    n_tokens = test_batch.n_tokens
                    test_loss = test_loss_compute(output, tgt_y, n_tokens)
            print ('train', train_iter_num, train_loss)
            print ('test', train_iter_num, test_loss)
            train_info = 'train,'+str(train_iter_num)+','+str(train_loss)
            test_info = 'test,'+str(train_iter_num)+','+str(test_loss)
            log_f.write(train_info+'\n')
            log_f.write(test_info+'\n')
            log_f.flush()
            model.train(True)
 
        if (train_iter_num % args.ckpt_interval == 0 or train_iter_num == args.total_iter) and dev_rank == 0:
            ckpt_path = os.path.join(args.ckpt_dir, 'ckpt.'+str(train_iter_num)+'.pt')
            print (ckpt_path)
            torch.save(model.state_dict(), ckpt_path)
            if train_iter_num == args.total_iter:
                break

if __name__ == '__main__':
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    np.random.seed(1111)
    argparser = argparse.ArgumentParser('training translation model')
    argparser.add_argument('--gpus', default='-1', type=str, help='gpu id')
    argparser.add_argument('--N', default=12, type=int, help='enc/dec layers')
    argparser.add_argument('--dim-model', default=256, type=int, help='model dim')
    argparser.add_argument('--dataset', default='file.txt', help='dataset')
    argparser.add_argument('--batch', default=16, type=int, help='batch size')
    argparser.add_argument('--workers-per-loader', default=2, type=int, help='loaders per worker')
    argparser.add_argument('--total-iter', default=1e6, type=int, help='total number of iterations')
    argparser.add_argument('--log-interval', default=100, type=int, help='log interval')
    argparser.add_argument('--ckpt-interval', default=100, type=int, help='ckpt interval')
    argparser.add_argument('--ckpt-dir', default='.', type=str, help='checkpoint path')
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
    if len(devices) == 1 and args.workers_per_loader == 0:
        args.ngpu = 0 if devices[0] < 0 else 1
        main(devices[0], args)
    else:
        try:
            args.ngpu = len(devices)
            mp = torch.multiprocessing.get_context('spawn')
            procs = []
            for dev_id in devices:
                procs.append(mp.Process(target=run, args=(dev_id, args),
                                        daemon=False))
                procs[-1].start()
            for p in procs:
                p.join()
        except KeyboardInterrupt:
            for p in procs:
                p.terminate()
                p.join()        
