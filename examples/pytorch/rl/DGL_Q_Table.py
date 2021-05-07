import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import dgl
import networkx as nx
import matplotlib.pyplot as plt

# Hyper Parameters
EPSILON = 0.5               # greedy policy
GAMMA = 0.95                 # reward discount
env = gym.make('FrozenLake-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_H = 20
BETA = 0.2
N_STATES = 16
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


class GNNLayer(nn.Module):
    def __init__(self):
        super(GNNLayer, self).__init__()

    def forward_from_record(self, g, h):
        return g.ndata['z'];
      
class GNN(nn.Module):
  def __init__(self, in_feats, hidden_size, num_classes):
      super(GNN, self).__init__()
      self.gnn = GNNLayer()

  def record(self, g, nodes_id, records):
        g.ndata['z'][nodes_id,:] = BETA * g.ndata['z'][nodes_id,:] + (1 - BETA) * records

  def forward(self, g, features):
        h = self.gnn.forward_from_record(g, features)
        return h
      
class DQN(object):
    def __init__(self):
        self.bg = dgl.DGLGraph()
        self.eval_net = GNN(N_STATES, N_H, N_ACTIONS)

    def add_nodes(self, features):
        nodes_id = self.bg.number_of_nodes()
        if nodes_id != 0:
            for i in range(len(self.bg.ndata['x'])):
                if self.bg.ndata['x'][i].equal(features[0]):
                    return i;
        self.bg.add_nodes(1, {'x': features, 'z': torch.zeros(1, N_ACTIONS)})
        src = [nodes_id]
        dst = [nodes_id]
        self.bg.add_edges(src, dst) 
        return nodes_id
        
    def choose_action(self, nodes_id):
        actions_value = self.eval_net(self.bg, self.bg.ndata['x'])[nodes_id]
        if np.random.uniform() < EPSILON:   # greedy
            action = torch.argmax(actions_value).data.item()
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
        Q = actions_value[action];

        return action, Q.detach().numpy()

    def learn_one(self, nodes_id, next_nodes_id, r):
        h_target = self.eval_net(self.bg, self.bg.ndata['x'])
        q_target = (r + GAMMA * h_target[next_nodes_id, :].max(0)[0])
        self.eval_net.record(self.bg, nodes_id, q_target)

dqn = DQN()
      
r_sum = 0
rList = []
for i in range(10000):
    s = env.reset()
    x = np.zeros((1, 16))
    x[0,s] = 1
    x = torch.FloatTensor(x)
    nodes_id = dqn.add_nodes(x)
    rAll = 0
    d = False
    j = 0
    while j < 99:
        j += 1
        a, Q = dqn.choose_action(nodes_id)
        s1, r, d, _ = env.step(a)
        x = np.zeros((1, 16))
        x[0,s1] = 1
        x = torch.FloatTensor(x)
        next_nodes_id = dqn.add_nodes(x)
        # Update Q_Table
        dqn.learn_one(nodes_id, next_nodes_id, r);
        rAll += r
        nodes_id = next_nodes_id
        if d == True:
            break
    rList.append(rAll)
    if (len(rList) == 1000):
        print("Score over time："+ str(sum(rList)/1000))
        rList = []
 
    
