# DGL Implementation of MVGRL
This DGL example implements the model proposed in the paper [Contrastive Multi-View Representation Learning on Graphs](https://arxiv.org/abs/2006.05582).

Paper link: https://arxiv.org/abs/2006.05582
Author's code: https://github.com/kavehhassani/mvgrl

## Example Implementor

This example was implemented by [Hengrui Zhang](https://github.com/hengruizhang98) when he was an applied scientist intern at AWS Shanghai AI Lab.

## Dependencies

- Python 3.7
- PyTorch 1.7.1
- dgl 0.6.0
- networkx
- scipy

## Datasets

##### Unsupervised Graph Classification Datasets:

 'MUTAG', 'PTC_MR', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K' of dgl.data.GINDataset.

| Dataset         | MUTAG | PTC_MR | RDT-B  | IMDB-B | IMDB-M |
| --------------- | ----- | ------ | ------ | ------ | ------ |
| # Graphs        | 188   | 344    | 2000   | 1000   | 1500   |
| # Classes       | 2     | 2      | 2      | 2      | 3      |
| Avg. Graph Size | 17.93 | 14.29  | 429.63 | 19.77  | 13.00  |

##### Unsupervised Node Classification Datasets:

'Cora', 'Citeseer' and 'Pubmed'

| Dataset  | # Nodes | # Edges | # Classes |
| -------- | ------- | ------- | --------- |
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |


## Arguments

##### 	Graph Classification:

```
--dataname         str     The graph dataset name.                Default is 'MUTAG'.
--gpu              int     GPU index.                             Default is -1, using cpu.
--epochs           int     Number of training periods.            Default is 200.
--batch_size       int     Size of a training batch.              Default is 64.
--patience         int     Early stopping steps.                  Default is 20.
--lr               float   Learning rate.           			  Default is 0.001.
--wd               float   Weight decay.            			  Default is 0.0.
--hid_dim          float   Embedding dimension.                   Default is 32.
--n_layers         int     Number of GNN layers.                  Default is 4
```

##### 	Node Classification:

```
--dataname         str     The graph dataset name.                Default is 'cora'.
--gpu              int     GPU index.                             Default is -1, using cpu.
--epochs           int     Number of training periods.            Default is 20.
--batch_size       int     Size of a training batch.              Default is 128.
--patience         int     Early stopping steps.                  Default is 20.
--lr1              float   Learning rate of main model.           Default is 0.001.
--lr2              float   Learning rate of linear classifer.     Default is 0.01.
--wd1              float   Weight decay of main model.            Default is 0.0.
--wd2			   float   Weight decay of linear classifier.     Default is 0.0.
--hid_dim          float   Embedding dimension.                   Default is 512.
```

## How to run examples

###### Graph Classification

```python
# Enter the 'graph' directory
cd graph

# MUTAG:
python main.py --dataname MUTAG --epochs 20

# PTC_MR:
python main.py --dataname PTC_MR --epochs 32 --hid_dim 128

# IMDB-BINARY
python main.py --dataname IMDB-BINARY --epochs 20 --hid_dim 512 --n_layers 2

# IMDB-MULTI
python main.py --dataname IMDB-MULTI --epochs 20 --hid_dim 512 --n_layers 2
```
###### Node Classification

For semi-supervised node classification on 'Cora', 'Citeseer' and 'Pubmed', we provide 2 implementions:

1. full-graph training, see 'main.py', where we contrast the local and global representations of the whole graph.
2. subgraph training, see 'main_sample.py', where we contrast the local and global representations of a sampled subgraph with fixed number of nodes.

For larger graphs(e.g. Pubmed), it would be hard to calculate the graph diffusion matrix(i.e, PPR matrix), so we try to approximate it with [APPNP](https://arxiv.org/abs/1810.05997), see funtion 'process_dataset_appnp'  in 'node/dataset.py' for details.

```python
# Enter the 'node' directory
cd node

# Cora with full graph
python main.py --dataname cora --gpu 0

# Cora with sampled subgraphs
python main_sample.py --dataname cora --gpu 0

# Citeseer with full graph
python main.py --dataname citeseer --wd1 0.001 --wd2 0.01 --epochs 200 --gpu 0

# Citeseer with sampled subgraphs
python main_sample.py --dataname citeseer --wd2 0.01 --gpu 0

# Pubmed with sampled subgraphs
python main_sample.py --dataname pubmed --sample_size 4000 --epochs 400 --patience 999 --gpu 0
```

## 	Performance

We use the same  hyper-parameter settings as stated in the original paper.

##### Graph Classification:

|      Dataset      | MUTAG | PTC-MR |  REDDIT-B  | IMDB-B | IMDB-M |
| :---------------: | :---: | :----: | :--------: | :----: | :----: |
| Accuracy Reported | 89.7  |  62.5  |    84.5    |  74.2  |  51.2  |
|        DGL        | 89.4  |  62.2  | run failed |  73.8  |  51.1  |

* The datasets that the authors used are slightly different from standard TUDatset (see dgl.data.GINDataset) in the nodes' feature dimensions. (e.g. The node features of 'MUTAG' dataset are 11 dimensional vectors instead of 7)
* We failed to load REDDIT-BINARY dataset with authors' code.

##### Node classification:

|      Dataset      | Cora | Citeseer | Pubmed |
| :---------------: | :--: | :------: | :----: |
| Accuracy Reported | 86.8 |   73.3   |  80.1  |
|    DGL-sample     | 83.2 |   72.6   |  79.8  |
|     DGL-full      | 83.5 |   73.7   |  OOM   |

* We faied to reproduce the reported accuracy on 'Cora', even with the authors' codes.
