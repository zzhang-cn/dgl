import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from rec.model.pinsage import PinSage
from rec.datasets.movielens import MovieLens
from rec.utils import cuda
from dgl import DGLGraph

import pickle
import os

cache_file = 'ml.pkl'

if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        ml = pickle.load(f)
else:
    ml = MovieLens('./ml-1m')
    with open(cache_file, 'wb') as f:
        pickle.dump(ml, f)

g = ml.g
neighbors = ml.user_neighbors + ml.movie_neighbors

n_hidden = 100
n_layers = 3
batch_size = 32

# Use the prior graph to train on user-product pairs in the training set.
# Validate on validation set.
# Note that each user-product pair is counted twice, but I think it is OK
# since we can treat product negative sampling and user negative sampling
# ubiquitously.

model = cuda(PinSage(g.number_of_nodes(), [n_hidden] * n_layers, 10, 5, 5))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)


def forward(model, g_prior, nodeset, train=True):
    if train:
        return model(g_prior, nodeset)
    else:
        with torch.no_grad():
            return model(g_prior, nodeset)


def filter_nid(nids, nid_from):
    nids = [nid.numpy() for nid in nids]
    nid_from = nid_from.numpy()
    np_mask = np.logical_and(*[np.isin(nid, nid_from) for nid in nids])
    return [torch.from_numpy(nid[np_mask]) for nid in nids]


def runtrain(g_prior_edges, g_train_edges, train):
    if train:
        model.train()
    else:
        model.eval()

    g_prior_src, g_prior_dst = g.find_edges(g_prior_edges)
    g_prior = DGLGraph()
    g_prior.add_nodes(g.number_of_nodes())
    g_prior.add_edges(g_prior_src, g_prior_dst)
    edge_batches = g_train_edges[torch.randperm(g_train_edges.shape[0])].split(batch_size)

    with tqdm.tqdm(edge_batches) as tq:
        sum_loss = 0
        sum_acc = 0
        count = 0
        for batch_id, batch in enumerate(tq):
            count += batch.shape[0]
            src, dst = g.find_edges(batch)
            dst_neg = []
            for i in range(len(dst)):
                nb = neighbors[dst[i].item()]
                mask = ~(g.has_edges_between(nb, src[i].item()).byte())
                dst_neg.append(np.random.choice(nb[mask].numpy()))
            dst_neg = torch.LongTensor(dst_neg)

            mask = (g_prior.in_degrees(dst_neg) > 0) & \
                   (g_prior.in_degrees(dst) > 0) & \
                   (g_prior.in_degrees(src) > 0)
            src = src[mask]
            dst = dst[mask]
            dst_neg = dst_neg[mask]
            if len(src) == 0:
                continue

            nodeset = cuda(torch.cat([src, dst, dst_neg]))
            src_size, dst_size, dst_neg_size = \
                    src.shape[0], dst.shape[0], dst_neg.shape[0]

            h_src, h_dst, h_dst_neg = (
                    forward(model, g_prior, nodeset, train)
                    .split([src_size, dst_size, dst_neg_size]))

            diff = (h_src * (h_dst_neg - h_dst)).sum(1)
            loss = (diff + 1).clamp(min=0).mean()
            acc = (diff < 0).sum()
            assert loss.item() == loss.item()

            if train:
                opt.zero_grad()
                loss.backward()
                for name, p in model.named_parameters():
                    assert (p.grad != p.grad).sum() == 0
                opt.step()

            sum_loss += loss.item()
            sum_acc += acc.item()
            avg_loss = sum_loss / (batch_id + 1)
            avg_acc = sum_acc / count
            tq.set_postfix({'loss': '%.6f' % loss.item(),
                            'avg_loss': avg_loss,
                            'avg_acc': avg_acc})

    return avg_loss, avg_acc


def runtest(g_prior_edges, validation=True):
    model.eval()

    n_users = len(ml.users.index)
    n_items = len(ml.movies.index)

    g_prior_src, g_prior_dst = g.find_edges(g_prior_edges)
    g_prior = DGLGraph()
    g_prior.add_nodes(g.number_of_nodes())
    g_prior.add_edges(g_prior_src, g_prior_dst)

    hs = []
    with torch.no_grad():
        with tqdm.trange(n_users + n_items) as tq:
            for node_id in tq:
                nodeset = cuda(torch.LongTensor([node_id]))
                h = forward(model, g_prior, nodeset, False)
                hs.append(h)
    h = torch.cat(hs, 0)

    rr = []

    with torch.no_grad():
        with tqdm.trange(n_users) as tq:
            for u_nid in tq:
                uid = ml.user_ids[u_nid]
                pids_exclude = ml.ratings[
                        (ml.ratings['user_id'] == uid) &
                        (ml.ratings['train'] | ml.ratings['test' if validation else 'valid'])
                        ]['movie_id'].values
                pids_candidate = ml.ratings[
                        (ml.ratings['user_id'] == uid) &
                        ml.ratings['valid' if validation else 'test']]['movie_id'].values
                pids = set(ml.movie_ids) - set(pids_exclude)
                p_nids = np.array([ml.movie_ids_invmap[pid] for pid in pids])
                p_nids_candidate = np.array([ml.movie_ids_invmap[pid] for pid in pids_candidate])

                dst = torch.from_numpy(p_nids)
                src = torch.zeros_like(dst).fill_(u_nid)
                h_dst = h[dst]
                h_src = h[src]

                score = (h_src * h_dst).sum(1)
                score_sort_idx = score.sort(descending=True)[1].cpu().numpy()

                is_candidate = np.isin(p_nids[score_sort_idx], p_nids_candidate)
                rank_candidates = is_candidate.nonzero()[0]
                rank = rank_candidates.min() + 1
                rr.append(rank)
                tq.set_postfix({'rank': rank})

    mrr = sum(1 / r for r in rr) / len(rr)
    return mrr


def train():
    best_mrr = 0
    logfile = open('output.log', 'w')
    for epoch in range(500):
        ml.refresh_mask()
        g_prior_edges = g.filter_edges(lambda edges: edges.data['prior'])
        g_train_edges = g.filter_edges(lambda edges: edges.data['train'] & ~edges.data['inv'])
        g_prior_train_edges = g.filter_edges(
                lambda edges: edges.data['prior'] | edges.data['train'])

        print('Epoch %d validation' % epoch)
        with torch.no_grad():
            valid_mrr = runtest(g_prior_train_edges, True)
            if best_mrr < valid_mrr:
                best_mrr = valid_mrr
                torch.save(model.state_dict(), 'model.pt')
        print('Epoch %d validation mrr:', valid_mrr)
        print('Epoch %d validation mrr:', valid_mrr, file=logfile)
        print('Epoch %d test' % epoch)
        with torch.no_grad():
            test_mrr = runtest(g_prior_train_edges, False)
        print('Epoch %d test mrr:', test_mrr)
        print('Epoch %d test mrr:', test_mrr, file=logfile)
        print('Epoch %d train' % epoch)
        runtrain(g_prior_edges, g_train_edges, True)
    logfile.close()


if __name__ == '__main__':
    train()
