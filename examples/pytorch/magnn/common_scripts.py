import time
import numpy as np
import argparse
import random
import torch.nn.functional as F
import torch.sparse
import dgl
from common_scripts import *

import torch.nn as nn
import torch.nn.functional as F
import scipy
from dgl.nn.pytorch import edge_softmax
import dgl.function as fn

import os, pickle, sys
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

class MLP(nn.Module):
    def __init__(self, n_layers, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        # W_r for each node
        self.n_layers = n_layers;
        if self.n_layers == 0:
            self.i2o = nn.Linear(in_dim, out_dim);
        else:
            self.i2h = nn.Linear(in_dim, hidden_dim);
            self.h2o = nn.Linear(hidden_dim, out_dim);
            self.h2h = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(self.n_layers-1)]);

    def forward(self, features):
        if self.n_layers == 0:
            return self.i2o(features);
        h = self.i2h(features);
        h = F.elu(h);
        for i in range(self.n_layers-1):
            h = self.h2h[i](h);
            h = F.elu(h);
        return self.h2o(h);
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss

class features_extractor_class:
    def __init__(self, device, features):
        self.device = device;
        self.features = features;
    def extract_features(self, block, which="src"):
        block_features = {}
        if which=="src":
            for ntype in block.srctypes:
                nid = block.srcnodes[ntype].data[dgl.NID]
                if ntype in self.features:
                    block_features[ntype] = self.features[ntype][nid].to(self.device)
        elif which=="dst":
            for ntype in block.dsttypes:
                nid = block.dstnodes[ntype].data[dgl.NID]
                if ntype in self.features:
                    block_features[ntype] = self.features[ntype][nid].to(self.device)
                    
        return block_features;
    def features_tensor(self, indices_tensor, type_tensor):
        features_list = [];
        for i in range(len(type_tensor)):
            features_list.append(self.features[type_tensor[i]][indices_tensor[:,i]].to(self.device))
                    
        return features_list;

def evaluate(net, block_list, label_list, class_weights, device):
    net.eval()
    preds_list = [];
    loss_list = [];
    for i, blocks in enumerate(block_list):
        logits, embeddings = net(blocks)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp, label_list[i], weight=class_weights)
        preds = logits.argmax(1).cpu().numpy();
        preds_list.append(preds);
        loss_list.append(loss.detach().cpu().item()*len(label_list[i]));
    all_labels = np.hstack([x.cpu().numpy() for x in label_list]);
    macro_f1 = f1_score(all_labels, np.hstack(preds_list), average='macro')
    micro_f1 = f1_score(all_labels, np.hstack(preds_list), average='micro')
    return np.sum(loss_list)/len(all_labels), macro_f1, micro_f1

def get_preds(net, block_list, label_list, device):
    net.eval()
    preds_list = [];
    logits_list = [];
    loss_list = [];
    for i, blocks in enumerate(block_list):
        logits, embeddings = net(blocks)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp, label_list[i])
        preds = logits.argmax(1).cpu().numpy();
        preds_list.append(preds);
        logits_list.append(logits.detach().cpu().numpy());
        loss_list.append(loss.detach().cpu().item()*len(label_list[i]));
    return np.vstack(logits_list), np.hstack(preds_list);

def get_splits(num_pts, seed, folds):
    indices = [i for i in range(num_pts)];
    random.seed(seed);
    random.shuffle(indices);
    indices = np.array(indices);
    start_index = 0;
    remaining_indices = num_pts;
    remaining_splits = folds;
    data_folds = [];
    for i in range(folds):
        end_index = start_index + remaining_indices//remaining_splits;
        data_folds.append(indices[start_index:end_index]);
        remaining_indices = remaining_indices - (end_index - start_index);
        remaining_splits = remaining_splits - 1;
        start_index = end_index;
    return data_folds;