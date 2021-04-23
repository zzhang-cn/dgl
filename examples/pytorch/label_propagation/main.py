import argparse
import torch
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from model import LabelPropagation


def main():
    # check cuda
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'

    # load data
    if args.dataset == 'Cora':
        dataset = CoraGraphDataset()
    elif args.dataset == 'Citeseer':
        dataset = CiteseerGraphDataset()
    elif args.dataset == 'Pubmed':
        dataset = PubmedGraphDataset()
    else:
        raise ValueError('Dataset {} is invalid.'.format(args.dataset))
    
    g = dataset[0]
    g = dgl.add_self_loop(g)

    labels = g.ndata.pop('label').to(device).long()

    # load masks for train / test, valid is not used.
    train_mask = g.ndata.pop('train_mask')
    test_mask = g.ndata.pop('test_mask')

    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze().to(device)
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze().to(device)

    g = g.to(device)
    
    # label propagation
    lp = LabelPropagation(args.num_layers, args.alpha)
    logits = lp(g, labels, mask=train_idx)

    test_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Acc {:.4f}".format(test_acc))


if __name__ == '__main__':
    """
    Label Propagation Hyperparameters
    """
    parser = argparse.ArgumentParser(description='LP')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--num-layers', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)

    args = parser.parse_args()
    print(args)

    main()
