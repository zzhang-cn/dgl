import time
import argparse
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader
import networkx as nx
import dgl

from models import MLP, InteractionNet, PrepareLayer
from dataloader import MultiBodyGraphCollator, MultiBodyTrainDataset,\
    MultiBodyValidDataset, MultiBodyTestDataset

from utils import make_video

def train(optimizer, loss_fn,reg_fn, model, prep, dataloader, lambda_reg, device):
    total_loss = 0
    model.train()
    for i, (graph_batch, data_batch, label_batch) in enumerate(dataloader):
        graph_batch = graph_batch.to(device)
        data_batch = data_batch.to(device)
        label_batch = label_batch.to(device)
        optimizer.zero_grad()
        node_feat, edge_feat = prep(graph_batch, data_batch)
        dummy_relation = torch.zeros(edge_feat.shape[0], 1).float().to(device)
        dummy_global = torch.zeros(node_feat.shape[0], 1).float().to(device)
        v_pred, out_e = model(graph_batch, node_feat[:, 3:5].float(
        ), edge_feat.float(), dummy_global, dummy_relation)
        loss = loss_fn(v_pred, label_batch)
        total_loss += float(loss)
        zero_target = torch.zeros_like(out_e)
        loss = loss + lambda_reg*reg_fn(out_e, zero_target)
        reg_loss = 0
        for param in model.parameters():
            reg_loss = reg_loss + lambda_reg * \
                reg_fn(param, torch.zeros_like(
                    param).float().to(device))
        loss = loss + reg_loss
        loss.backward()
        optimizer.step()
    return total_loss/(i+1)

# One step evaluation


def eval(loss_fn, model, prep, dataloader, device):
    total_loss = 0
    model.eval()
    for i, (graph_batch, data_batch, label_batch) in enumerate(dataloader):
        graph_batch = graph_batch.to(device)
        data_batch = data_batch.to(device)
        label_batch = label_batch.to(device)
        node_feat, edge_feat = prep(graph_batch, data_batch)
        dummy_relation = torch.zeros(
            edge_feat.shape[0], 1).float().to(device)
        dummy_global = torch.zeros(
            node_feat.shape[0], 1).float().to(device)
        v_pred, _ = model(graph_batch, node_feat[:, 3:5].float(
        ), edge_feat.float(), dummy_global, dummy_relation)
        loss = loss_fn(v_pred, label_batch)
        total_loss += float(loss)
    return total_loss/(i+1)

# Rollout Evaluation based in initial state
# Need to integrate


def eval_rollout(model, prep, initial_frame, n_object, device):
    current_frame = initial_frame.to(device)
    base_graph = nx.complete_graph(n_object)
    graph = dgl.from_networkx(base_graph).to(device)
    pos_buffer = []
    model.eval()
    for step in range(100):
        node_feats, edge_feats = prep(graph, current_frame)
        dummy_relation = torch.zeros(
            edge_feats.shape[0], 1).float().to(device)
        dummy_global = torch.zeros(
            node_feats.shape[0], 1).float().to(device)
        v_pred, _ = model(graph, node_feats[:, 3:5].float(
        ), edge_feats.float(), dummy_global, dummy_relation)
        current_frame[:, [1, 2]] += v_pred*0.001
        current_frame[:, 3:5] = v_pred
        pos_buffer.append(current_frame[:, [1, 2]].cpu().numpy())
    pos_buffer = np.vstack(pos_buffer).reshape(100, n_object, -1)
    make_video(pos_buffer, 'video_model.mp4')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--lr', type=float, default=0.001,
                           help='learning rate')
    argparser.add_argument('--epochs', type=int, default=40000,
                           help='Number of epochs in training')
    argparser.add_argument('--lambda_reg', type=float, default=0.001,
                           help='regularization weight')
    argparser.add_argument('--gpu', type=int, default=-1,
                           help='gpu device code, -1 means cpu')
    argparser.add_argument('--batch_size', type=int, default=100,
                           help='size of each mini batch')
    argparser.add_argument('--num_workers', type=int, default=0,
                           help='number of workers for dataloading')
    argparser.add_argument('--visualize', action='store_true', default=False,
                           help='Whether enable trajectory rollout mode for visualization')
    args = argparser.parse_args()

    # Select Device to be CPU or GPU
    if args.gpu != -1:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    train_data = MultiBodyTrainDataset()
    valid_data = MultiBodyValidDataset()
    test_data = MultiBodyTestDataset()
    collator = MultiBodyGraphCollator(train_data.n_particles)

    train_dataloader = DataLoader(
        train_data, args.batch_size, True, collate_fn=collator, num_workers=args.num_workers)
    valid_dataloader = DataLoader(
        valid_data, args.batch_size, True, collate_fn=collator, num_workers=args.num_workers)
    test_full_dataloader = DataLoader(
        test_data, args.batch_size, True, collate_fn=collator, num_workers=args.num_workers)

    node_feats = 5
    stat = {'median': torch.from_numpy(train_data.stat_median).to(device),
            'max': torch.from_numpy(train_data.stat_max).to(device),
            'min': torch.from_numpy(train_data.stat_min).to(device)}
    print("Weight: ", train_data.stat_median[0],
          train_data.stat_max[0], train_data.stat_min[0])
    print("Position: ", train_data.stat_median[[
          1, 2]], train_data.stat_max[[1, 2]], train_data.stat_min[[1, 2]])
    print("Velocity: ", train_data.stat_median[[
          3, 4]], train_data.stat_max[[3, 4]], train_data.stat_min[[3, 4]])

    prepare_layer = PrepareLayer(node_feats, stat).to(device)
    interaction_net = InteractionNet(node_feats, stat).to(device)
    print(interaction_net)
    optimizer = torch.optim.Adam(interaction_net.parameters(), lr=args.lr)
    state_dict = interaction_net.state_dict()

    loss_fn = torch.nn.MSELoss()
    reg_fn = torch.nn.MSELoss(reduction='sum')
    try:
        for e in range(args.epochs):
            last_t = time.time()
            loss = train(optimizer, loss_fn,reg_fn, interaction_net,
                         prepare_layer, train_dataloader, args.lambda_reg, device)
            print("Epoch time: ", time.time()-last_t)
            if e % 1 == 0:
                valid_loss = eval(loss_fn, interaction_net,
                                  prepare_layer, valid_dataloader, device)
                test_full_loss = eval(
                    loss_fn, interaction_net, prepare_layer, test_full_dataloader, device)
                print("Epoch: {}.Loss: Valid: {} Full: {}".format(
                    e, valid_loss, test_full_loss))
    except:
        traceback.print_exc()
    finally:
        if args.visualize:
            eval_rollout(interaction_net, prepare_layer,
                         test_data.first_frame, test_data.n_particles, device)
            make_video(test_data.test_traj[:100, :, [1, 2]], 'video_truth.mp4')
