import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from sampler import SAINTNodeSampler, SAINTEdgeSampler, SAINTRandomWalkSampler
from modules import GCNNet
from utils import Logger, evaluate, save_log_dir, load_data


def main(args):

    multilabel_data = set(['ppi', 'yelp', 'amazon'])
    multilabel = args.dataset in multilabel_data

    # load and preprocess dataset
    data = load_data(args, multilabel)
    g = data.g
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    labels = g.ndata['label']

    train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)

    in_feats = g.ndata['feat'].shape[1]
    n_classes = data.num_classes
    n_nodes = g.num_nodes()
    n_edges = g.num_edges()

    n_train_samples = train_mask.int().sum().item()
    n_val_samples = val_mask.int().sum().item()
    n_test_samples = test_mask.int().sum().item()

    print("""----Data statistics------'
    #Nodes %d
    #Edges %d
    #Classes %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
          (n_nodes, n_edges, n_classes,
           n_train_samples,
           n_val_samples,
           n_test_samples))
    # load sampler
    if args.sampler == "node":
        subg_iter = SAINTNodeSampler(args.node_budget, args.dataset, g,
                                     train_nid, args.num_repeat)
    elif args.sampler == "edge":
        subg_iter = SAINTEdgeSampler(args.edge_budget, args.dataset, g,
                                     train_nid, args.num_repeat)
    elif args.sampler == "rw":
        subg_iter = SAINTRandomWalkSampler(args.num_roots, args.length, args.dataset, g,
                                            train_nid, args.num_repeat)

    # set device for dataset tensors
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        g = g.to(args.gpu)

    print('labels shape:', g.ndata['label'].shape)
    print("features shape, ", g.ndata['feat'].shape)

    model = GCNNet(
        in_dim=in_feats,
        hid_dim=args.n_hidden,
        out_dim=n_classes,
        arch=args.arch,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        aggr=args.aggr
    )

    if cuda:
        model.cuda()

    # logger and so on
    log_dir = save_log_dir(args)
    logger = Logger(os.path.join(log_dir, 'loggings'))
    logger.write(args)

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # set train_nids to cuda tensor
    if cuda:
        train_nid = torch.from_numpy(train_nid).cuda()
        print("current memory after model before training",
              torch.cuda.memory_allocated(device=train_nid.device) / 1024 / 1024)
    start_time = time.time()
    best_f1 = -1

    for epoch in range(args.n_epochs):
        for j, subg in enumerate(subg_iter):
            # sync with upper level training graph
            if cuda:
                subg = subg.to(torch.cuda.current_device())
            model.train()
            # forward
            pred = model(subg)
            batch_labels = subg.ndata['label']

            if multilabel:
                loss = F.binary_cross_entropy_with_logits(pred, batch_labels, reduction='sum',
                                                          weight=subg.ndata['l_n'].unsqueeze(1))
            else:
                loss = F.cross_entropy(pred, batch_labels, reduction='none')
                loss = (subg.ndata['l_n'] * loss).sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 5)
            optimizer.step()
            if j == len(subg_iter) - 1:
                print(f"epoch:{epoch+1}/{args.n_epochs}, Iteration {j+1}/"
                      f"{len(subg_iter)}:training loss", loss.item())

        # evaluate
        if epoch % args.val_every == 0:
            val_f1_mic, val_f1_mac = evaluate(
                model, g, labels, val_mask, multilabel)
            print(
                "Val F1-mic {:.4f}, Val F1-mac {:.4f}".format(val_f1_mic, val_f1_mac))
            if val_f1_mic > best_f1:
                best_f1 = val_f1_mic
                print('new best val f1:', best_f1)
                torch.save(model.state_dict(), os.path.join(
                    log_dir, 'best_model.pkl'))

    end_time = time.time()
    print(f'training using time {end_time - start_time}')

    # test
    if True:
        model.load_state_dict(torch.load(os.path.join(
            log_dir, 'best_model.pkl')))
    test_f1_mic, test_f1_mac = evaluate(
        model, g, labels, test_mask, multilabel)
    print("Test F1-mic {:.4f}, Test F1-mac {:.4f}".format(test_f1_mic, test_f1_mac))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    # data source params
    parser.add_argument("--dataset", type=str, choices=['ppi', 'flickr', 'reddit', 'yelp', 'amazon'], default='ppi',
                        help="Name of dataset.")

    # cuda params
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU index. Default: -1, using CPU.")

    # sampler params
    parser.add_argument("--sampler", type=str, default="node", choices=['node', 'edge', 'rw'],
                        help="Type of sampler")
    parser.add_argument("--node-budget", type=int, default=6000,
                        help="Expected number of sampled nodes when using node sampler")
    parser.add_argument("--edge-budget", type=int, default=4000,
                        help="Expected number of sampled edges when using edge sampler")
    parser.add_argument("--num-roots", type=int, default=3000,
                        help="Expected number of sampled root nodes when using random walk sampler")
    parser.add_argument("--length", type=int, default=2,
                        help="The length of random walk when using random walk sampler")
    parser.add_argument("--num-repeat", type=int, default=50,
                        help="Number of repeating sampling one node to estimate edge / node probability")

    # model params
    parser.add_argument("--n-hidden", type=int, default=512,
                        help="Number of hidden gcn units")
    parser.add_argument("--arch", type=str, default="1-0-1-0",
                        help="Network architecture. 1 means an order-1 layer (self feature plus 1-hop neighbor "
                             "feature), and 0 means an order-0 layer (self feature only)")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout rate")
    parser.add_argument("--batch-norm", action='store_true',
                        help="Whether to use batch norm")
    parser.add_argument("--aggr", type=str, default="concat", choices=['mean', 'concat'],
                        help="How to aggregate the self feature and neighbor features")

    # training params
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0,
                        help="Weight for L2 reg")
    parser.add_argument("--val-every", type=int, default=1,
                        help="Frequency of evaluation on the validation set in number of epochs")
    parser.add_argument("--note", type=str, default='none',
                        help="Note for log dir")

    args = parser.parse_args()

    print(args)

    main(args)
