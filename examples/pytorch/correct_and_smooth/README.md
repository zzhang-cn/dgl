# DGL Implementation of CorrectAndSmooth

This DGL example implements the GNN model proposed in the paper [Combining Label Propagation and Simple Models Out-performs Graph Neural Networks](https://arxiv.org/abs/2010.13993). For the original implementation, see [here](https://github.com/CUAI/CorrectAndSmooth).

Contributor: [xnuohz](https://github.com/xnuohz)

### Requirements
The codebase is implemented in Python 3.7. For version requirement of packages, see below.

```
dgl 0.6.0.post1
torch 1.7.0
ogb 1.3.0
```

### The graph datasets used in this example

Open Graph Benchmark(OGB). Dataset summary:

|    Dataset    |  #Nodes   |   #Edges   | #Node Feats |  Metric  |
| :-----------: | :-------: | :--------: | :---------: | :------: |
|  ogbn-arxiv   |  169,343  | 1,166,243  |     128     | Accuracy |
| ogbn-products | 2,449,029 | 61,859,140 |     100     | Accuracy |

### Usage

Training a **Base predictor** and using **Correct&Smooth** which follows the original hyperparameters on different datasets.

##### ogbn-arxiv

* **MLP + C&S**

```bash
python main.py
python main.py --pretrain
```

* **Linear + C&S**

```bash
python main.py --model linear
python main.py --model linear --pretrain --correction-alpha 0.8 --smoothing-alpha 0.6
```

* **GAT + C&S**

```bash
python main.py --model gat --epochs 2000 --lr 0.002 --dropout 0.75
python main.py --model gat --pretrain
```

##### ogbn-products

* **Linear + C&S**

```bash
python main.py --dataset ogbn-products --model linear
python main.py --dataset ogbn-products --model linear --pretrain --correction-alpha 0.6 --smoothing-alpha 0.9
```

### Performance

#### ogbn-arxiv

|                 |  MLP  | MLP + C&S | Linear | Linear + C&S |  GAT  | GAT + C&S |
| :-------------: | :---: | :-------: | :----: | :----------: | :---: | :-------: |
| Results(Author) | 55.58 |   68.72   | 51.06  |    70.24     | 73.28 |   73.64   |
|  Results(DGL)   | 55.06 |   69.75   | 51.06  |    70.14     | 72.02 |   72.79   |

#### ogbn-products

|                 |  MLP  | MLP + C&S | Linear | Linear + C&S |
| :-------------: | :---: | :-------: | :----: | :----------: |
| Results(Author) | 60.22 |   83.75   | 47.67  |    82.34     |
|  Results(DGL)   |  OOM  |    OOM    | 47.76  |    79.27     |