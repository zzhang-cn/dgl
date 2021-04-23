# DGL Implementation of the TAHIN

This DGL example implements the TAHIN module proposed in the paper [HCDIR](https://arxiv.org/pdf/2007.15293.pdf). Since the code and dataset have not been published yet, we implement its main idea and emperiment on two other datasets.

Example implementor
----------------------
This example was implemented by [KounianhuaDu](https://github.com/KounianhuaDu) during her software development intern time at the AWS Shanghai AI Lab.

Dependencies
----------------------
- pytorch 1.7.1
- dgl 0.6.0

Datasets
---------------------------------------
The datasets used can be downloaded from [here](https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding). For the experiments, all the positive edges are fetched and the same number of negative edges are randomly sampled. The edges are then shuffled and splitted into train/validate/test at a ratio of 6:2:2. The positive edges that appear in the validation and test sets are then removed from the original graph.

The original graph statistics:

**Movielens** 

(Source : https://grouplens.org/datasets/movielens/)

| Entity         |#Entity        |
| :-------------:|:-------------:|
| User           | 943           |
| Age            | 8             |
| Occupation     | 21            |
| Movie          | 1,682         |
| Genre          | 18            |

| Relation            |#Relation      |
| :-------------:     |:-------------:|
| User - Movie        | 100,000       |
| User - User (KNN)   | 47,150        |
| User - Age          | 943           |
| User - Occupation   | 943           |
| Movie - Movie (KNN) | 82,798        |
| Movie - Genre       | 2,861         |

**Amazon** 

(Source : http://jmcauley.ucsd.edu/data/amazon/)

| Entity         |#Entity        |
| :-------------:|:-------------:|
| User           | 6,170         |
| Item           | 2,753         |
| View           | 3,857         |           
| Category       | 22            |
| Brand          | 334           |

| Relation          |#Relation      |
| :-------------:   |:-------------:|
| User - Item       | 195,791       |
| Item - View       | 5,694         |
| Item - Category   | 5,508         | 
| Item - Brand      | 2,753         |

How to run
--------------------------------

```python
python main.py --dataset amazon --gpu 0
```


```python
python main.py --dataset movielens --gpu 0
```


Performance
-------------------------
**Results**

| Dataset |         Movielens        |          Amazon          |
|---------| ------------------------ | ------------------------ |
|  Metric |    HAN     /    TAHIN    |    HAN     /    TAHIN    |
|   AUC   |   0.9297   /   0.9392    |   0.8470   /   0.8442    |
|   ACC   |   0.8627   /   0.8683    |   0.7672   /   0.7619    |
|    F1   |   0.8631   /   0.8707    |   0.7628   /   0.7499    |
| Logloss |   0.3689   /   0.3266    |   0.5311   /   0.5150    |
