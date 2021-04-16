# DGL Implementation of JKNet

This DGL example implements the GNN model proposed in the paper [Representation Learning on Graphs with Jumping Knowledge Networks](https://arxiv.org/abs/1806.03536).

Contributor: [xnuohz](https://github.com/xnuohz)

### Requirements
The codebase is implemented in Python 3.6. For version requirement of packages, see below.

```
dgl 0.6.0
scikit-learn 0.24.1
tqdm 4.56.0
torch 1.7.1
```

### The graph datasets used in this example

###### Node Classification

The DGL's built-in Cora, Citeseer datasets. Dataset summary:

| Dataset | #Nodes | #Edges | #Feats | #Classes | #Train Nodes | #Val Nodes | #Test Nodes |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Cora | 2,708 | 10,556 | 1,433 | 7(single label) | 60% | 20% | 20% |
| Citeseer | 3,327 | 9,228 | 3,703 | 6(single label) | 60% | 20% | 20% |

### Usage

###### Dataset options
```
--dataset          str     The graph dataset name.             Default is 'Cora'.
```

###### GPU options
```
--gpu              int     GPU index.                          Default is -1, using CPU.
```

###### Model options
```
--run              int     Number of running times.                    Default is 10.
--epochs           int     Number of training epochs.                  Default is 500.
--lr               float   Adam optimizer learning rate.               Default is 0.01.
--lamb             float   L2 regularization coefficient.              Default is 0.0005.
--hid-dim          int     Hidden layer dimensionalities.              Default is 32.
--num-layers       int     Number of T.                                Default is 5.
--mode             str     Type of aggregation ['cat', 'max', 'lstm']. Default is 'cat'.
--dropout          float   Dropout applied at all layers.              Default is 0.5.
```

###### Examples

The following commands learn a neural network and predict on the test set.
Train a JKNet which follows the original hyperparameters on different datasets.
```bash
# Cora:
python main.py --gpu 0 --mode max --num-layers 6
python main.py --gpu 0 --mode cat --num-layers 6
python main.py --gpu 0 --mode lstm --num-layers 1

# Citeseer:
python main.py --gpu 0 --dataset Citeseer --mode max --num-layers 1
python main.py --gpu 0 --dataset Citeseer --mode cat --num-layers 1
python main.py --gpu 0 --dataset Citeseer --mode lstm --num-layers 2
```

### Performance

**As the author does not release the code, we don't have the access to the data splits they used.**

###### Node Classification

* Cora

|  | JK-Maxpool | JK-Concat | JK-LSTM |
| :-: | :-: | :-: | :-: |
| Metrics(Table 2) | 89.6±0.5 | 89.1±1.1 | 85.8±1.0 |
| Metrics(DGL) | 86.1±1.5 | 85.1±1.6 | 84.2±1.6 |

* Citeseer

|  | JK-Maxpool | JK-Concat | JK-LSTM |
| :-: | :-: | :-: | :-: |
| Metrics(Table 2) | 77.7±0.5 | 78.3±0.8 | 74.7±0.9 |
| Metrics(DGL) | 70.9±1.9 | 73.0±1.5 | 69.0±1.7 |