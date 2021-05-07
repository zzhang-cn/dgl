import torch
import torch.nn.functional as F
import dgl
import tqdm
import numpy as np
from collections import Counter
from model import *
from ladies2 import *

def compute_acc(pred, label):
    return (pred.argmax(1) == label).float().mean()

def train(g, n_classes, args):
    in_feats = g.ndata['features'].shape[1]
    device = args['device']
    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]

    conv = {
            'gcn': GraphConv,
            'sage': SAGEConv,
            }[args['conv']]
    norm = {
            'gcn': normalized_laplacian_edata,
            'sage': normalized_edata,
            }[args['conv']]
    g.edata['weight'] = norm(g)

    num_nodes = [int(n) for n in args['num_nodes'].split(',')]
    sampler = LADIESNeighborSampler(num_nodes, weight='weight', out_weight='w', replace=False)
    #sampler = LADIESNeighborSampler(g, num_nodes, weight='weight', out_weight='w', replace=False)
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=args['num_workers'])
    val_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        val_nid,
        sampler,
        batch_size=args['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=args['num_workers'])

    model = Model(in_feats, args['hidden_dim'], n_classes, len(num_nodes),
                  dropout=args['dropout'], conv=conv)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args['lr'])
    best_acc = 0

    for epoch in range(args['num_epochs']):
        model.train()
        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, seeds, blocks) in enumerate(tq):
                blocks = [block.to(device) for block in blocks]
                batch_inputs = blocks[0].srcdata['features']
                batch_labels = blocks[-1].dstdata['labels']

                batch_pred = model(blocks, batch_inputs)
                loss = F.cross_entropy(batch_pred, batch_labels)
                acc = compute_acc(batch_pred, batch_labels)

                opt.zero_grad()
                loss.backward()
                opt.step()

                tq.set_postfix({'loss': '%.06f' % loss.item(), 'acc': '%.03f' % acc.item()})

        model.eval()
        val_pred = []
        val_labels = []
        with torch.no_grad():
            with tqdm.tqdm(val_dataloader) as tq:
                for step, (input_nodes, seeds, blocks) in enumerate(tq):
                    blocks = [block.to(device) for block in blocks]
                    batch_inputs = blocks[0].srcdata['features']
                    batch_labels = blocks[-1].dstdata['labels']

                    batch_pred = model(blocks, batch_inputs)
                    val_pred.append(batch_pred)
                    val_labels.append(batch_labels)
            acc = compute_acc(torch.cat(val_pred, 0), torch.cat(val_labels, 0)).item()

        print('Best: %.4f Val: %.4f' % (best_acc, acc))
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), 'model.pt')

    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    pred = model.inference(g, g.ndata['features'], g.edata['weight'], args['batch_size'], args['device'], args['num_workers'])
    val_pred = pred[val_nid]
    val_label = g.ndata['labels'][val_nid]
    test_pred = pred[test_nid]
    test_label = g.ndata['labels'][test_nid]
    print('Val Acc', compute_acc(val_pred, val_label))
    print('Test Acc', compute_acc(test_pred, test_label))

def load_ogb(name):
    from ogb.nodeproppred import DglNodePropPredDataset

    print('load', name)
    data = DglNodePropPredDataset(name=name)
    print('finish loading', name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata['features'] = graph.ndata['feat']
    graph.ndata['labels'] = labels
    in_feats = graph.ndata['features'].shape[1]
    num_labels = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    train_mask[train_nid] = True
    val_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    val_mask[val_nid] = True
    test_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    print('finish constructing', name)
    return graph, num_labels


if __name__ == '__main__':
    g, num_labels = load_ogb('ogbn-products')
    #g = dgl.remove_self_loop(g)
    #g = dgl.add_self_loop(g)

    args = {
        'num_epochs': 50,
        'num_workers': 4,
        'batch_size': 1000,
        'hidden_dim': 256,
        'lr': 0.001,
        'num_nodes': '5000,5000,5000',
        'device': 'cuda:0',
        'conv': 'sage',
        'dropout': 0.}
    train(g, num_labels, args)
