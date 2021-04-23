# DGL Implementation of LabelPropagation

This DGL example implements the method proposed in the paper [Learning from Labeled and Unlabeled Data with Label Propagation](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.3864&rep=rep1&type=pdf).

Contributor: [xnuohz](https://github.com/xnuohz)

### Requirements
The codebase is implemented in Python 3.7. For version requirement of packages, see below.

```
dgl 0.6.0.post1
torch 1.7.0
```

### The graph datasets used in this example

The DGL's built-in Cora, Pubmed and Citeseer datasets. Dataset summary:

| Dataset  | #Nodes | #Edges | #Feats | #Classes | #Train Nodes | #Val Nodes | #Test Nodes |
| :------: | :----: | :----: | :----: | :------: | :----------: | :--------: | :---------: |
| Citeseer | 3,327  | 9,228  | 3,703  |    6     |     120      |    500     |    1000     |
|   Cora   | 2,708  | 10,556 | 1,433  |    7     |     140      |    500     |    1000     |
|  Pubmed  | 19,717 | 88,651 |  500   |    3     |      60      |    500     |    1000     |

### Usage

```bash
# Cora
python main.py

# Citeseer
python main.py --dataset Citeseer

# Pubmed
python main.py --dataset Pubmed --num-layers 60 --alpha 1
```

### Performance

|   Dataset    | Cora  | Citeseer | Pubmed |
| :----------: | :---: | :------: | :----: |
| Results(DGL) | 68.00 |  51.40   | 69.20  |
