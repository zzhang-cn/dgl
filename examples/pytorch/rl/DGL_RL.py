import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import dgl
import networkx as nx
import matplotlib.pyplot as plt
from gym.envs.registration import register

# Hyper Parameters
EPSILON = 0.5               # greedy policy
GAMMA = 0.95                 # reward discount
env = gym.make('FrozenLake-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
BETA = 0.2
N_STATES = 16
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


class GNNLayer(nn.Module):
    def __init__(self):
        super(GNNLayer, self).__init__()

    def message_func(self, edges):
        Q = GAMMA * torch.max(edges.src['z'], dim = -1, keepdim = True)[0] * edges.data['e'][:,1,:] + edges.data['e'][:,0,:]
        a_count = edges.data['e'][:,1,:]
        return {'q': Q, 'ac' : a_count}

    def reduce_func(self, nodes):
        z = BETA * nodes.data['z'] + (1 - BETA) * torch.sum(nodes.mailbox['q'], dim = 1) / (torch.sum(nodes.mailbox['ac'], dim = 1) + 1e-6)
        return {'z': z}

    def bp(self, g):
        g.update_all(self.message_func, self.reduce_func)

    def forward(self, g):
        return g.ndata['z']

    def forward_from_record(self, g, h):
        return g.ndata['z']
      
class GNN(nn.Module):
  def __init__(self):
      super(GNN, self).__init__()
      self.gnn = GNNLayer()

  def record(self, g, nodes_id, records):
      g.ndata['z'][nodes_id,:] = BETA * g.ndata['z'][nodes_id,:] + (1 - BETA) * records

  def bp(self, g):
      self.gnn.bp(g)

  def forward(self, g):
      h = self.gnn(g)
      return h
      
class DQN(object):
    def __init__(self):
        self.bg = dgl.DGLGraph()
        self.eval_net = GNN()

    def add_edges(self, nodes_id, next_nodes_id, a, r):
        if nodes_id == next_nodes_id:
            return
        src = [nodes_id]
        dst = [next_nodes_id]
        if self.bg.has_edge_between(next_nodes_id, nodes_id):
            edge = torch.zeros([1, 2, N_ACTIONS])
            edge[0, 0, a] = r
            edge[0, 1, a] = 1.0
            self.bg.edges[next_nodes_id, nodes_id].data['e'] += edge
            return
        edge = torch.zeros([1, 2, N_ACTIONS])
        edge[0, 0, a] = r
        edge[0, 1, a] = 1.0
        #print(edge)
        self.bg.add_edges(dst, src, {'e': edge})

    def add_nodes(self, features):
        nodes_id = self.bg.number_of_nodes()
        if nodes_id != 0:
            for i in range(len(self.bg.ndata['x'])):
                if self.bg.ndata['x'][i].equal(features[0]):
                    return i;
        self.bg.add_nodes(1, {'x': features, 'z': torch.zeros(1, N_ACTIONS)})
        return nodes_id
        
    def choose_action(self, nodes_id):
        actions_value = self.eval_net(self.bg)[nodes_id]
        if np.random.uniform() < EPSILON:   # greedy
            action = torch.argmax(actions_value).data.item()
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
        Q = actions_value[action];

        return action, Q.detach().numpy()

    def learn(self):
        self.eval_net.bp(self.bg)

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
        dqn.add_edges(nodes_id, next_nodes_id, a, r)
        # Update Q_Table
        rAll += r
        nodes_id = next_nodes_id
        if d == True:
            for i in range(20):
                if rAll != 0:
                    dqn.learn()
            break
    rList.append(rAll)
    if (len(rList) == 1000):
        print("Score over time："+ str(sum(rList)/1000))
        rList = []
