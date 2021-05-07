# Reinforcement Learning(RL) with Graph Neural Network(GNN)
We do an RL with GNN demo based on DGL

## Introduction

This demo has two parts:

1）We implement the Q-learning algorithm with Q-Table on DGL。

2）We design a new RL mechanism based on GNN. It uses a graph to record the states and the transformation among them and update the Q value with the message passing mechanism.

### Environment

[FrozenLake-v0](https://gym.openai.com/envs/FrozenLake-v0/)：it is from OpenAI Gym，which has 16 states.

### Q-Learning Algorithm
Q-learning is a off-policy TD control algorithm, defined by

$$Q(S_t, A_t) = Q(S_t, A_t) + \alpha[R_{t+1} + \gamma\max_a Q(S_{t+1}, a) - Q(S_t, A_t)].$$

### Install
```bash
pip install dgl
pip install pytorch
pip install gym
```

## Demo 1 : Q-Table with GNN
First, we construct a graph,
```bash
self.bg = dgl.DGLGraph().
```
Second, we use add_nodes function to add new nodes. In the graph, the 'x' represents the feature of a node and the 'z' represents the Q value of a node. Therefore, a Q-Table is represented by a graph. In this function, we just add a node with new features which is stored in 'x'. 
```bash
def add_nodes(self, features):
    nodes_id = self.bg.number_of_nodes()
    if nodes_id != 0:
        for i in range(len(self.bg.ndata['x'])):
            if self.bg.ndata['x'][i].equal(features[0]):
                return i;
    self.bg.add_nodes(1, {'x': features, 'z': torch.zeros(1, N_ACTIONS)})
```
Then we iteratively updates the Q-Table by Q-learning algorithem.
```bash
def learn_one(self, nodes_id, next_nodes_id, r):
    h_target = self.eval_net(self.bg, self.bg.ndata['x'])
    q_target = (r + GAMMA * h_target[next_nodes_id, :].max(0)[0])
    self.eval_net.record(self.bg, nodes_id, q_target)
```
At last, we predict the action under a state by the Q value in each node of the graph.
```bash
def forward_from_record(self, g):
    return g.ndata['z'];
```
## Demo 2 : RL with GNN
In Demo 2, instead of using the Q-learning algorithm, we use a graph to construct a perfect model of the environment. Given the model, we compute the optimal policy by message passing function in GNN. In our demo, the policy improvement equation is defined by

$$ Q_{i+1}(s, a) = \beta * Q_i(s, a) + (1 - \beta)\sum_{s',r}p(s',r|s,a)[r + \gamma max_{a'}Q_i(s',a')].$$

As a consequense, we need to construct a graph with nodes and edges. 
We use nodes to represent the states and use the edges to represent the relationship between nodes. There are three properties in an edge from node A to node B：

- A is the next state of B.
- The first column records the reward from B to A
- The second column records the number of actions that we take when we encounter B and then step into A in the game.

```bash
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
     self.bg.add_edges(dst, src, {'e': edge})
```
Besides, we design a new graph message passing mechanism to learn the Q value.
- In message_fuc, we send the product of the maximum Q value of node B and the number of actions, and the number of actions from B to A. Therefore, we calculate
the $n(A, a, B, r) * \gamma * maxQ(B, a') + r$ and $n(A, a, B, r)$ and send it from B to A. 
- In reduce_fuc, we collect each Q value under the each action and the total numbers of each action. We calculate the new $Q(s, a)$ by $\sum_{s'}\frac{n(s, a, s',r)}{\sum_\hat{a} n(s, \hat{a}, s', r)} * \gamma * maxQ(s', a')$. The $\frac{n(s, a, s', r)}{\sum_\hat{a} n(s, \hat{a}, s', r)}$ is used to evaluate the $p(s', r| s, a)$ 
```bash
 def message_func(self, edges):
     Q = GAMMA * torch.max(edges.src['z'], dim = -1, keepdim = True)[0] \
       * edges.data['e'][:,1,:] + edges.data['e'][:,0,:]
     a_count = edges.data['e'][:,1,:]
     return {'q': Q, 'ac' : a_count}

 def reduce_func(self, nodes):
     z = BETA * nodes.data['z'] + (1 - BETA) * \ 
            torch.sum(nodes.mailbox['q'],  dim = 1)\
            / (torch.sum(nodes.mailbox['ac'], dim = 1) + 1e-6)
     return {'z': z}
```

# Evaluation：
We compare the performance of Q-Table, Q-Table with GNN (Demo 1) and RL with GNN (Demo2) on the FrozenLake-v0

| Epoch| Q-Table | Q-Table with GNN | RL with GNN|
| ------ | ------ | ------ |------|
| 1 | 0.029 | 0.013|0.054|
| 2 | 0.040 | 0.006 |0.077|
| 3 | 0.032 | 0.009 |0.072|
| 4 | 0.039 | 0.012 |0.085|
| 5 | 0.040 | 0.005|0.075|
| 6 | 0.029 | 0.007 |0.075|
| 7 | 0.030 | 0.015|0.070|
| 8 | 0.034 | 0.006 |0.064|
| 9 | 0.032 | 0.008 |0.060|
| 10 | 0.035 | 0.009 |0.070|

## License
[MIT](https://choosealicense.com/licenses/mit/)
