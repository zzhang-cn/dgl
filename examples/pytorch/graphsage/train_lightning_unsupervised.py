import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import dgl.function as fn
import time
import argparse
import tqdm
import glob
import os

from negative_sampler import NegativeSampler

from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from model import SAGE, compute_acc_unsupervised as compute_acc
from load_graph import load_reddit, inductive_split, load_ogb

class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        score = th.cat([pos_score, neg_score])
        label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)]).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss

class SAGELightning(LightningModule):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 lr):
        super().__init__()
        self.save_hyperparameters()
        self.module = SAGE(in_feats, n_hidden, n_classes, n_layers, activation, dropout)
        self.lr = lr
        self.loss_fcn = CrossEntropyLoss()
        self.acc = Accuracy()

    def training_step(self, batch, batch_idx):
        input_nodes, pos_graph, neg_graph, mfgs = batch
        mfgs = [mfg.int().to(device) for mfg in mfgs]
        pos_graph = pos_graph.to(device)
        neg_graph = neg_graph.to(device)
        batch_inputs = mfgs[0].srcdata['features']
        batch_labels = mfgs[-1].dstdata['labels']
        batch_pred = self.module(mfgs, batch_inputs)
        loss = self.loss_fcn(batch_pred, pos_graph, neg_graph)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_nodes, output_nodes, mfgs = batch
        mfgs = [mfg.int().to(device) for mfg in mfgs]
        batch_inputs = mfgs[0].srcdata['features']
        batch_labels = mfgs[-1].dstdata['labels']
        batch_pred = self.module(mfgs, batch_inputs)
        return batch_pred

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class DataModule(LightningDataModule):
    def __init__(self, dataset_name, data_cpu=False, fan_out=[10, 25],
                 device=th.device('cpu'), batch_size=1000, num_workers=4):
        super().__init__()
        if dataset_name == 'reddit':
            g, n_classes = load_reddit()
            n_edges = g.num_edges()
            reverse_eids = th.cat([
                th.arange(n_edges // 2, n_edges),
                th.arange(0, n_edges // 2)])
        elif dataset_name == 'ogbn-products':
            g, n_classes = load_ogb('ogbn-products')
            n_edges = g.num_edges()
            # The reverse edge of edge 0 in OGB products dataset is 1.
            # The reverse edge of edge 2 is 3.  So on so forth.
            reverse_eids = th.arange(n_edges) ^ 1
        else:
            raise ValueError('unknown dataset')

        train_nid = th.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
        val_nid = th.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
        test_nid = th.nonzero(~(g.ndata['train_mask'] | g.ndata['val_mask']), as_tuple=True)[0]

        sampler = dgl.dataloading.MultiLayerNeighborSampler([int(_) for _ in fan_out])

        dataloader_device = th.device('cpu')
        if not data_cpu:
            train_nid = train_nid.to(device)
            val_nid = val_nid.to(device)
            test_nid = test_nid.to(device)
            g = g.formats(['csc'])
            g = g.to(device)
            dataloader_device = device

        self.g = g
        self.train_nid, self.val_nid, self.test_nid = train_nid, val_nid, test_nid
        self.sampler = sampler
        self.device = dataloader_device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_feats = g.ndata['features'].shape[1]
        self.n_classes = n_classes
        self.reverse_eids = reverse_eids

    def train_dataloader(self):
        return dgl.dataloading.EdgeDataLoader(
            self.g,
            np.arange(self.g.num_edges()),
            self.sampler,
            exclude='reverse_id',
            reverse_eids=self.reverse_eids,
            negative_sampler=NegativeSampler(self.g, args.num_negs, args.neg_share),
            device=self.device,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers)

    def val_dataloader(self):
        # Note that the validation data loader is a NodeDataLoader
        # as we want to evaluate all the node embeddings.
        return dgl.dataloading.NodeDataLoader(
            self.g,
            np.arange(self.g.num_nodes()),
            self.sampler,
            device=self.device,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers)


class UnsupervisedClassification(Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_outputs = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.val_outputs.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        node_emb = th.cat(self.val_outputs, 0)
        g = trainer.datamodule.g
        labels = g.ndata['labels']
        f1_micro, f1_macro = compute_acc(
            node_emb, labels, trainer.datamodule.train_nid,
            trainer.datamodule.val_nid, trainer.datamodule.test_nid)
        pl_module.log('val_f1_micro', f1_micro)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument("--gpu", type=int, default=0)
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--num-negs', type=int, default=1)
    argparser.add_argument('--neg-share', default=False, action='store_true',
                           help="sharing neg nodes for positive nodes")
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=10000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=1000)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    datamodule = DataModule(
        args.dataset, True, [int(_) for _ in args.fan_out.split(',')],
        device, args.batch_size, args.num_workers)
    model = SAGELightning(
        datamodule.in_feats, args.num_hidden, datamodule.n_classes, args.num_layers,
        F.relu, args.dropout, args.lr)

    # Train
    unsupervised_callback = UnsupervisedClassification()
    checkpoint_callback = ModelCheckpoint(monitor='val_f1_micro', save_top_k=1)
    trainer = Trainer(gpus=[args.gpu] if args.gpu != -1 else None,
                      max_epochs=args.num_epochs,
                      val_check_interval=1000,
                      callbacks=[checkpoint_callback, unsupervised_callback],
                      num_sanity_val_steps=0)
    trainer.fit(model, datamodule=datamodule)
