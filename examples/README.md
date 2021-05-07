# Official DGL Examples and Modules

The folder contains example implementations of selected research papers related to Graph Neural Networks. Note that the examples may not work with incompatible DGL versions.
* For examples working with the latest master (or the latest [nightly build](https://www.dgl.ai/pages/start.html)), check out https://github.com/dmlc/dgl/tree/master/examples.
* For examples working with a certain release, check out `https://github.com/dmlc/dgl/tree/<release_version>/examples` (E.g., https://github.com/dmlc/dgl/tree/0.5.x/examples)

## Overview

| Paper                                                        | node classification | link prediction / classification | graph property prediction | sampling           | OGB                |
| ------------------------------------------------------------ | ------------------- | -------------------------------- | ------------------------- | ------------------ | ------------------ |
| [Latent Dirichlet Allocation](#lda) | :heavy_check_mark:  |    :heavy_check_mark:  |  |  |  |
| [Network Embedding with Completely-imbalanced Labels](#rect) | :heavy_check_mark:  |                                  |                           |                    |                    |
| [Boost then Convolve: Gradient Boosting Meets Graph Neural Networks](#bgnn) | :heavy_check_mark:  |                                  |                           |                    |                    |
| [Contrastive Multi-View Representation Learning on Graphs](#mvgrl) | :heavy_check_mark: |  | :heavy_check_mark: |  |  |
| [Deep Graph Contrastive Representation Learning](#grace) | :heavy_check_mark: | |  | | |
| [Graph Random Neural Network for Semi-Supervised Learning on Graphs](#grand) | :heavy_check_mark:  |                                  |                           |                    |                    |
| [Heterogeneous Graph Transformer](#hgt)                      | :heavy_check_mark:  | :heavy_check_mark:               |                           |                    |                    |
| [Graph Convolutional Networks for Graphs with Multi-Dimensionally Weighted Edges](#mwe) | :heavy_check_mark:  |                                  |                           |                    | :heavy_check_mark: |
| [SIGN: Scalable Inception Graph Neural Networks](#sign)      | :heavy_check_mark:  |                                  |                           |                    | :heavy_check_mark: |
| [Strategies for Pre-training Graph Neural Networks](#prestrategy) |                     |                                  | :heavy_check_mark:        |                    |                    |
| [InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization](#infograph) | | | :heavy_check_mark: | | |
| [Graph Neural Networks with convolutional ARMA filters](#arma) | :heavy_check_mark:  |                                  |                           |                    |                    |
| [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](#appnp) | :heavy_check_mark:  |                                  |                           |                    |                    |
| [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](#clustergcn) | :heavy_check_mark:  |                                  |                           | :heavy_check_mark: | :heavy_check_mark: |
| [Deep Graph Infomax](#dgi)                                   | :heavy_check_mark:  |                                  |                           |                    |                    |
| [Hierarchical Graph Representation Learning with Differentiable Pooling](#diffpool) |                     |                                  | :heavy_check_mark:        |                    |                    |
| [Representation Learning for Attributed Multiplex Heterogeneous Network](#gatne-t) |                     | :heavy_check_mark:               |                           |                    |                    |
| [How Powerful are Graph Neural Networks?](#gin)              | :heavy_check_mark:  |                                  | :heavy_check_mark:        |                    | :heavy_check_mark: |
| [Heterogeneous Graph Attention Network](#han)                | :heavy_check_mark:  |                                  |                           |                    |                    |
| [Simplifying Graph Convolutional Networks](#sgc)             | :heavy_check_mark:  |                                  |                           |                    |                    |
| [Molecular Property Prediction: A Multilevel Quantum Interactions Modeling Perspective](#mgcn) |                     |                                  | :heavy_check_mark:        |                    |                    |
| [Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism](#attentivefp) |                     |                                  | :heavy_check_mark:        |                    |                    |
| [MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing](#mixhop) | :heavy_check_mark:  |                                  |                           |                    |                    |
| [Graph Attention Networks](#gat)                             | :heavy_check_mark:  |                                  |                           |                    | :heavy_check_mark: |
| [Attention-based Graph Neural Network for Semi-supervised Learning](#agnn) | :heavy_check_mark:  |                                  |                           | :heavy_check_mark: |                    |
| [Graph Convolutional Neural Networks for Web-Scale Recommender Systems](#pinsage) |                     |                                  |                           |                    |                    |
| [Semi-Supervised Classification with Graph Convolutional Networks](#gcn) | :heavy_check_mark:  | :heavy_check_mark:               | :heavy_check_mark:        |                    | :heavy_check_mark: |
| [Graph Convolutional Matrix Completion](#gcmc)               |                     | :heavy_check_mark:               |                           |                    |                    |
| [Inductive Representation Learning on Large Graphs](#graphsage) | :heavy_check_mark:  | :heavy_check_mark:               |                           | :heavy_check_mark: | :heavy_check_mark: |
| [metapath2vec: Scalable Representation Learning for Heterogeneous Networks](#metapath2vec) | :heavy_check_mark:  |                                  |                           |                    |                    |
| [Topology Adaptive Graph Convolutional Networks](#tagcn)     | :heavy_check_mark:  |                                  |                           |                    |                    |
| [Modeling Relational Data with Graph Convolutional Networks](#rgcn) | :heavy_check_mark:  | :heavy_check_mark:               |                           | :heavy_check_mark: |                    |
| [Neural Message Passing for Quantum Chemistry](#mpnn)        |                     |                                  | :heavy_check_mark:        |                    |                    |
| [SchNet: A continuous-filter convolutional neural network for modeling quantum interactions](#schnet) |                     |                                  | :heavy_check_mark:        |                    |                    |
| [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](#chebnet) | :heavy_check_mark:  |                                  | :heavy_check_mark:        |                    |                    |
| [Geometric deep learning on graphs and manifolds using mixture model CNNs](#monet) | :heavy_check_mark:  |                                  | :heavy_check_mark:        |                    |                    |
| [Molecular Graph Convolutions: Moving Beyond Fingerprints](#weave) |                     |                                  | :heavy_check_mark:        |                    |                    |
| [LINE: Large-scale Information Network Embedding](#line)     |                     | :heavy_check_mark:               |                           |                    | :heavy_check_mark: |
| [DeepWalk: Online Learning of Social Representations](#deepwalk) |                     | :heavy_check_mark:               |                           |                    | :heavy_check_mark: |
| [Self-Attention Graph Pooling](#sagpool)                     |                     |                                  | :heavy_check_mark:        |                    |                    |
| [Convolutional Networks on Graphs for Learning Molecular Fingerprints](#nf) |                     |                                  | :heavy_check_mark:        |                    |                    |
| [GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation](#gnnfilm) | :heavy_check_mark:  |                     |                     |                     |                     |
| [Hierarchical Graph Pooling with Structure Learning](#hgp-sl)                                                                                 |                     |                                  | :heavy_check_mark:        |                    |                    |
| [Graph Representation Learning via Hard and Channel-Wise Attention Networks](#hardgat)                                   |:heavy_check_mark:                     |                 |                           |                    |                    |
| [Neural Graph Collaborative Filtering](#ngcf) |             | :heavy_check_mark: |                     |                   |                   |
| [Graph Cross Networks with Vertex Infomax Pooling](#gxn)                                   |                     |                                  | :heavy_check_mark:        |                    |                    |
| [Towards Deeper Graph Neural Networks](#dagnn) | :heavy_check_mark:  |                                  |                           |                    |                    |
| [The PageRank Citation Ranking: Bringing Order to the Web](#pagerank) |  |                                  |                           |                    |                    |
| [Fast Suboptimal Algorithms for the Computation of Graph Edit Distance](#beam) |  |                                  |                           |                    |                    |
| [Speeding Up Graph Edit Distance Computation with a Bipartite Heuristic](#astar) |  |                                  |                           |                    |                    |
| [A Three-Way Model for Collective Learning on Multi-Relational Data](#rescal) |  |                                  |                           |                    |                    |
| [Speeding Up Graph Edit Distance Computation through Fast Bipartite Matching](#bipartite) |  |                                  |                           |                    |                    |
| [Translating Embeddings for Modeling Multi-relational Data](#transe) |  |                                  |                           |                    |                    |
| [A Hausdorff Heuristic for Efficient Computation of Graph Edit Distance](#hausdorff) |  |                                  |                           |                    |                    |
| [Embedding Entities and Relations for Learning and Inference in Knowledge Bases](#distmul) |  |                                  |                           |                    |                    |
| [Learning Entity and Relation Embeddings for Knowledge Graph Completion](#transr) |  |                                  |                           |                    |                    |
| [Order Matters: Sequence to sequence for sets](#seq2seq) |                     |                                  | :heavy_check_mark:        |                    |                    |
| [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](#treelstm) |  |                                  |                           |                    |                    |
| [Complex Embeddings for Simple Link Prediction](#complex) |  |                                  |                           |                    |                    |
| [Gated Graph Sequence Neural Networks](#ggnn) |  |                                  |                           |                    |                    |
| [Atomic Convolutional Networks for Predicting Protein-Ligand Binding Affinity](#acnn) |  |                                  |                           |                    |                    |
| [Attention Is All You Need](#transformer) |  |                                  |                           |                    |                    |
| [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](#pointnet++) |  |                                  |                           |                    |                    |
| [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](#pointnet) |  |                                  |                           |                    |                    |
| [Dynamic Routing Between Capsules](#capsule) |  |                                  |                           |                    |                    |
| [An End-to-End Deep Learning Architecture for Graph Classification](#dgcnn) |                     |                                  | :heavy_check_mark:        |                    |                    |
| [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](#stgcn) |  |                                  |                           |                    |                    |
| [Recurrent Relational Networks](#rrn) |  |                                  |                           |                    |                    |
| [Junction Tree Variational Autoencoder for Molecular Graph Generation](#jtvae) |  |                                  |                           |                    |                    |
| [Learning Deep Generative Models of Graphs](#dgmg) |  |                                  |                           |                    |                    |
| [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](#rotate) |  |                                  |                           |                    |                    |
| [A graph-convolutional neural network model for the prediction of chemical reactivity](#wln) |  |                                  |                           |                    |                    |
| [Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks](#settrans) |                     |                                  | :heavy_check_mark:        |                    |                    |
| [Graphical Contrastive Losses for Scene Graph Parsing](#scenegraph) |  |                                  |                           |                    |                    |
| [Dynamic Graph CNN for Learning on Point Clouds](#dgcnnpoint) |  |                                  |                           |                    |                    |
| [Supervised Community Detection with Line Graph Neural Networks](#lgnn) |  |                                  |                           |                    |                    |
| [Text Generation from Knowledge Graphs with Graph Transformers](#graphwriter) |  |                                  |                           |                    |                    |
| [Temporal Graph Networks For Deep Learning on Dynamic Graphs](#tgn) |  | :heavy_check_mark:                                 |                           |                    |                    |
| [Directional Message Passing for Molecular Graphs](#dimenet) |  |                                  | :heavy_check_mark: |                           |                    |
| [Link Prediction Based on Graph Neural Networks](#seal) | | :heavy_check_mark: | | :heavy_check_mark: | :heavy_check_mark: |
| [Variational Graph Auto-Encoders](#vgae) |  | :heavy_check_mark: | | | |
| [Composition-based Multi-Relational Graph Convolutional Networks](#compgcn)|  |  :heavy_check_mark: | | | |
| [GNNExplainer: Generating Explanations for Graph Neural Networks](#gnnexplainer) |  :heavy_check_mark: |                                  |                                  |                                  |                                  |
| [Interaction Networks for Learning about Objects, Relations and Physics](#graphsim) |  |                                 |:heavy_check_mark:                 |                    |                    |
| [Representation Learning on Graphs with Jumping Knowledge Networks](#jknet) |  :heavy_check_mark: |                                  |                                  |                                  |                                  |
| [DeeperGCN: All You Need to Train Deeper GCNs](#deepergcn)                                                                        |                                  |                                  | :heavy_check_mark:        |                    | :heavy_check_mark: |
| [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forcasting](#dcrnn) |                     |                                  | :heavy_check_mark:        |                    |                    |
| [GaAN: Gated Attention Networks for Learning on large and Spatiotemporal Graphs](#gaan) |                     |                                  | :heavy_check_mark:        |                    |                    |

## 2021

- <a name="bgnn"></a> Ivanov et al. Boost then Convolve: Gradient Boosting Meets Graph Neural Networks. [Paper link](https://openreview.net/forum?id=ebS5NUfoMKL). 
    - Example code: [PyTorch](../examples/pytorch/bgnn)
    - Tags: semi-supervised node classification, tabular data, GBDT

## 2020

- <a name="rect"></a> Wang et al. Network Embedding with Completely-imbalanced Labels. [Paper link](https://ieeexplore.ieee.org/document/8979355). 
    - Example code: [PyTorch](../examples/pytorch/rect)
    - Tags:  node classification, network embedding, completely-imbalanced labels
- <a name="mvgrl"></a> Hassani and Khasahmadi. Contrastive Multi-View Representation Learning on Graphs. [Paper link](https://arxiv.org/abs/2006.05582). 
    - Example code: [PyTorch](../examples/pytorch/mvgrl)
    - Tags: graph diffusion, self-supervised learning on graphs.
- <a name="grace"></a> Zhu et al. Deep Graph Contrastive Representation Learning. [Paper link](https://arxiv.org/abs/2006.04131). 
    - Example code: [PyTorch](../examples/pytorch/grace)
    - Tags: contrastive learning for node classification.
- <a name="grand"></a> Feng et al. Graph Random Neural Network for Semi-Supervised Learning on Graphs. [Paper link](https://arxiv.org/abs/2005.11079). 
    - Example code: [PyTorch](../examples/pytorch/grand)
    - Tags: semi-supervised node classification, simplifying graph convolution, data augmentation
- <a name="hgt"></a> Hu et al. Heterogeneous Graph Transformer. [Paper link](https://arxiv.org/abs/2003.01332).
    - Example code: [PyTorch](../examples/pytorch/hgt)
    - Tags: dynamic heterogeneous graphs, large-scale, node classification, link prediction
- <a name="mwe"></a> Chen. Graph Convolutional Networks for Graphs with Multi-Dimensionally Weighted Edges. [Paper link](https://cims.nyu.edu/~chenzh/files/GCN_with_edge_weights.pdf).
    - Example code: [PyTorch on ogbn-proteins](../examples/pytorch/ogb/ogbn-proteins)
    - Tags: node classification, weighted graphs, OGB
- <a name="sign"></a> Frasca et al. SIGN: Scalable Inception Graph Neural Networks. [Paper link](https://arxiv.org/abs/2004.11198).
    - Example code: [PyTorch on ogbn-arxiv/products/mag](../examples/pytorch/ogb/sign), [PyTorch](../examples/pytorch/sign)
    - Tags: node classification, OGB, large-scale, heterogeneous graphs
- <a name="prestrategy"></a> Hu et al. Strategies for Pre-training Graph Neural Networks. [Paper link](https://arxiv.org/abs/1905.12265).
    - Example code: [Molecule embedding](https://github.com/awslabs/dgl-lifesci/tree/master/examples/molecule_embeddings), [PyTorch for custom data](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/csv_data_configuration)
    - Tags: molecules, graph classification, unsupervised learning, self-supervised learning, molecular property prediction
- <a name="gnnfilm"></a> Marc Brockschmidt. GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation. [Paper link](https://arxiv.org/abs/1906.12192).
    - Example code: [PyTorch](../examples/pytorch/GNN-FiLM)
    - Tags: multi-relational graphs, hypernetworks, GNN architectures
- <a name="gxn"></a> Li, Maosen, et al. Graph Cross Networks with Vertex Infomax Pooling. [Paper link](https://arxiv.org/abs/2010.01804).
    - Example code: [PyTorch](../examples/pytorch/gxn)
    - Tags: pooling, graph classification
- <a name="dagnn"></a> Liu et al. Towards Deeper Graph Neural Networks. [Paper link](https://arxiv.org/abs/2007.09296).
    - Example code: [PyTorch](../examples/pytorch/dagnn)
    - Tags: over-smoothing, node classification
- <a name="dimenet"></a> Klicpera et al. Directional Message Passing for Molecular Graphs. [Paper link](https://arxiv.org/abs/2003.03123).
    - Example code: [PyTorch](../examples/pytorch/dimenet)
    - Tags: molecules, molecular property prediction, quantum chemistry
- <a name="tgn"></a> Rossi et al. Temporal Graph Networks For Deep Learning on Dynamic Graphs. [Paper link](https://arxiv.org/abs/2006.10637).
    - Example code: [Pytorch](../examples/pytorch/tgn)
    - Tags: over-smoothing, node classification 
- <a name="compgcn"></a> Vashishth, Shikhar, et al. Composition-based Multi-Relational Graph Convolutional Networks. [Paper link](https://arxiv.org/abs/1911.03082).
    - Example code: [PyTorch](../examples/pytorch/compGCN)
    - Tags: multi-relational graphs, graph neural network
- <a name="deepergcn"></a> Li et al. DeeperGCN: All You Need to Train Deeper GCNs. [Paper link](https://arxiv.org/abs/2006.07739).
    - Example code: [PyTorch](../examples/pytorch/deepergcn)
    - Tags: over-smoothing, deeper gnn, OGB

## 2019


- <a name="infograph"></a> Sun et al. InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization. [Paper link](https://arxiv.org/abs/1908.01000). 
    - Example code: [PyTorch](../examples/pytorch/infograph)
    - Tags: semi-supervised graph regression, unsupervised graph classification
- <a name="arma"></a>  Bianchi et al. Graph Neural Networks with Convolutional ARMA Filters. [Paper link](https://arxiv.org/abs/1901.01343).
    - Example code: [PyTorch](../examples/pytorch/arma)
    - Tags: node classification
- <a name="appnp"></a> Klicpera et al. Predict then Propagate: Graph Neural Networks meet Personalized PageRank. [Paper link](https://arxiv.org/abs/1810.05997).
    - Example code: [PyTorch](../examples/pytorch/appnp), [MXNet](../examples/mxnet/appnp)
    - Tags: node classification
- <a name="clustergcn"></a> Chiang et al. Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks. [Paper link](https://arxiv.org/abs/1905.07953).
    - Example code: [PyTorch](../examples/pytorch/cluster_gcn), [PyTorch-based GraphSAGE variant on OGB](../examples/pytorch/ogb/cluster-sage), [PyTorch-based GAT variant on OGB](../examples/pytorch/ogb/cluster-gat)
    - Tags: graph partition, node classification, large-scale, OGB, sampling
- <a name="dgi"></a> Veličković et al. Deep Graph Infomax. [Paper link](https://arxiv.org/abs/1809.10341).
    - Example code: [PyTorch](../examples/pytorch/dgi), [TensorFlow](../examples/tensorflow/dgi)
    - Tags: unsupervised learning, node classification
- <a name="diffpool"></a> Ying et al. Hierarchical Graph Representation Learning with Differentiable Pooling. [Paper link](https://arxiv.org/abs/1806.08804).
    - Example code: [PyTorch](../examples/pytorch/diffpool)
    - Tags: pooling, graph classification, graph coarsening
- <a name="gatne-t"></a> Cen et al. Representation Learning for Attributed Multiplex Heterogeneous Network. [Paper link](https://arxiv.org/abs/1905.01669v2).
    - Example code: [PyTorch](../examples/pytorch/GATNE-T)
    - Tags: heterogeneous graphs, link prediction, large-scale
- <a name="gin"></a> Xu et al. How Powerful are Graph Neural Networks? [Paper link](https://arxiv.org/abs/1810.00826).
    - Example code: [PyTorch on graph classification](../examples/pytorch/gin), [PyTorch on node classification](../examples/pytorch/model_zoo/citation_network), [PyTorch on ogbg-ppa](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/ogbg_ppa), [MXNet](../examples/mxnet/gin)
    - Tags: graph classification, node classification, OGB
- <a name="graphwriter"></a> Koncel-Kedziorski et al. Text Generation from Knowledge Graphs with Graph Transformers. [Paper link](https://arxiv.org/abs/1904.02342).
    - Example code: [PyTorch](../examples/pytorch/graphwriter)
    - Tags: knowledge graph, text generation
- <a name="han"></a> Wang et al. Heterogeneous Graph Attention Network. [Paper link](https://arxiv.org/abs/1903.07293).
    - Example code: [PyTorch](../examples/pytorch/han)
    - Tags: heterogeneous graphs, node classification
- <a name="lgnn"></a> Chen et al. Supervised Community Detection with Line Graph Neural Networks. [Paper link](https://arxiv.org/abs/1705.08415).
    - Example code: [PyTorch](../examples/pytorch/line_graph)
    - Tags: line graph, community detection
- <a name="sgc"></a> Wu et al. Simplifying Graph Convolutional Networks. [Paper link](https://arxiv.org/abs/1902.07153).
    - Example code: [PyTorch](../examples/pytorch/sgc), [MXNet](../examples/mxnet/sgc)
    - Tags: node classification
- <a name="dgcnnpoint"></a> Wang et al. Dynamic Graph CNN for Learning on Point Clouds. [Paper link](https://arxiv.org/abs/1801.07829).
    - Example code: [PyTorch](../examples/pytorch/pointcloud/edgeconv)
    - Tags: point cloud classification
- <a name="scenegraph"></a> Zhang et al. Graphical Contrastive Losses for Scene Graph Parsing. [Paper link](https://arxiv.org/abs/1903.02728).
    - Example code: [MXNet](../examples/mxnet/scenegraph)
    - Tags: scene graph extraction
- <a name="settrans"></a> Lee et al. Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. [Paper link](https://arxiv.org/abs/1810.00825).
    - Pooling module: [PyTorch encoder](https://docs.dgl.ai/api/python/nn.pytorch.html#settransformerencoder), [PyTorch decoder](https://docs.dgl.ai/api/python/nn.pytorch.html#settransformerdecoder)
    - Tags: graph classification
- <a name="wln"></a> Coley et al. A graph-convolutional neural network model for the prediction of chemical reactivity. [Paper link](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc04228d#!divAbstract).
    - Example code: [PyTorch](https://github.com/awslabs/dgl-lifesci/tree/master/examples/reaction_prediction/rexgen_direct)
    - Tags: molecules, reaction prediction
- <a name="mgcn"></a> Lu et al. Molecular Property Prediction: A Multilevel Quantum Interactions Modeling Perspective. [Paper link](https://arxiv.org/abs/1906.11081).
    - Example code: [PyTorch](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/alchemy)
    - Tags: molecules, quantum chemistry
- <a name="attentivefp"></a> Xiong et al. Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism. [Paper link](https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959).
    - Example code: [PyTorch (with attention visualization)](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/pubchem_aromaticity), [PyTorch for custom data](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/csv_data_configuration)
    - Tags: molecules, molecular property prediction
- <a name="rotate"></a> Sun et al. RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space. [Paper link](https://arxiv.org/pdf/1902.10197.pdf).
    - Example code: [PyTorch](https://github.com/awslabs/dgl-ke/tree/master/examples), [PyTorch for custom data](https://aws-dglke.readthedocs.io/en/latest/commands.html)
    - Tags: knowledge graph embedding
- <a name="mixhop"></a> Abu-El-Haija et al. MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing. [Paper link](https://arxiv.org/abs/1905.00067).
    - Example code: [PyTorch](../examples/pytorch/mixhop)
    - Tags: node classification
- <a name="sagpool"></a> Lee, Junhyun, et al. Self-Attention Graph Pooling. [Paper link](https://arxiv.org/abs/1904.08082).
    - Example code: [PyTorch](../examples/pytorch/sagpool)
    - Tags: graph classification, pooling
- <a name="hgp-sl"></a> Zhang, Zhen, et al. Hierarchical Graph Pooling with Structure Learning. [Paper link](https://arxiv.org/abs/1911.05954).
    - Example code: [PyTorch](../examples/pytorch/hgp_sl)
    - Tags: graph classification, pooling
- <a name='hardgat'></a> Gao, Hongyang, et al. Graph Representation Learning via Hard and Channel-Wise Attention Networks [Paper link](https://arxiv.org/abs/1907.04652).
    - Example code: [PyTorch](../examples/pytorch/hardgat)
    - Tags: node classification, graph attention
- <a name='ngcf'></a> Wang, Xiang, et al. Neural Graph Collaborative Filtering. [Paper link](https://arxiv.org/abs/1905.08108).
    - Example code: [PyTorch](../examples/pytorch/NGCF)
    - Tags: Collaborative Filtering, Recommendation, Graph Neural Network 
- <a name='gnnexplainer'></a> Ying, Rex, et al. GNNExplainer: Generating Explanations for Graph Neural Networks. [Paper link](https://arxiv.org/abs/1903.03894).
    - Example code: [PyTorch](../examples/pytorch/gnn_explainer)
    - Tags: Graph Neural Network, Explainability


## 2018

- <a name="dgmg"></a> Li et al. Learning Deep Generative Models of Graphs. [Paper link](https://arxiv.org/abs/1803.03324).
    - Example code: [PyTorch example for cycles](../examples/pytorch/dgmg), [PyTorch example for molecules](https://github.com/awslabs/dgl-lifesci/tree/master/examples/generative_models/dgmg)
    - Tags: generative models, autoregressive models, molecules

- <a name="gat"></a> Veličković et al. Graph Attention Networks. [Paper link](https://arxiv.org/abs/1710.10903).
    - Example code: [PyTorch](../examples/pytorch/gat), [PyTorch on ogbn-arxiv](../examples/pytorch/ogb/ogbn-arxiv), [PyTorch on ogbn-products](../examples/pytorch/ogb/ogbn-products), [TensorFlow](../examples/tensorflow/gat), [MXNet](../examples/mxnet/gat)
    - Tags: node classification, OGB

- <a name="jtvae"></a> Jin et al. Junction Tree Variational Autoencoder for Molecular Graph Generation. [Paper link](https://arxiv.org/abs/1802.04364).
    - Example code: [PyTorch](../examples/pytorch/jtnn)
    - Tags: generative models, molecules, VAE

- <a name="agnn"></a> Thekumparampil et al. Attention-based Graph Neural Network for Semi-supervised Learning. [Paper link](https://arxiv.org/abs/1803.03735).
    - Example code: [PyTorch](../examples/pytorch/model_zoo/citation_network)
    - Tags: node classification
    
- <a name="pinsage"></a> Ying et al. Graph Convolutional Neural Networks for Web-Scale Recommender Systems. [Paper link](https://arxiv.org/abs/1806.01973).
    - Example code: [PyTorch](../examples/pytorch/pinsage)
    - Tags: recommender system, large-scale, sampling

- <a name="rrn"></a> Berg Palm et al. Recurrent Relational Networks. [Paper link](https://arxiv.org/abs/1711.08028).
    - Example code: [PyTorch](../examples/pytorch/rrn)
    - Tags: sudoku solving

- <a name="stgcn"></a> Yu et al. Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting. [Paper link](https://arxiv.org/abs/1709.04875v4).
    - Example code: [PyTorch](../examples/pytorch/stgcn_wave)
    - Tags: spatio-temporal, traffic forecasting

- <a name="dgcnn"></a> Zhang et al. An End-to-End Deep Learning Architecture for Graph Classification. [Paper link](https://www.cse.wustl.edu/~ychen/public/DGCNN.pdf).
    - Pooling module: [PyTorch](https://docs.dgl.ai/api/python/nn.pytorch.html#sortpooling), [TensorFlow](https://docs.dgl.ai/api/python/nn.tensorflow.html#sortpooling), [MXNet](https://docs.dgl.ai/api/python/nn.mxnet.html#sortpooling)
    - Tags: graph classification

- <a name="seal"></a>  Zhang et al. Link Prediction Based on Graph Neural Networks. [Paper link](https://papers.nips.cc/paper/2018/file/53f0d7c537d99b3824f0f99d62ea2428-Paper.pdf).
    - Example code: [PyTorch](../examples/pytorch/seal)
    - Tags: link prediction, sampling

- <a name="jknet"></a>  Xu et al. Representation Learning on Graphs with Jumping Knowledge Networks. [Paper link](https://arxiv.org/abs/1806.03536).
    - Example code: [PyTorch](../examples/pytorch/jknet)
    - Tags: message passing, neighborhood

- <a name="gaan"></a> Zhang et al. GaAN: Gated Attention Networks for Learning on Large and Spatiotemporal Graphs. [Paper link](https://arxiv.org/abs/1803.07294).
    - Example code: [pytorch](../examples/pytorch/dtgrnn)
    - Tags: Static discrete temporal graph, traffic forcasting

## 2017

- <a name="gcn"></a> Kipf and Welling. Semi-Supervised Classification with Graph Convolutional Networks. [Paper link](https://arxiv.org/abs/1609.02907). 
    - Example code: [PyTorch](../examples/pytorch/gcn), [PyTorch on ogbn-arxiv](../examples/pytorch/ogb/ogbn-arxiv), [PyTorch on ogbl-ppa](https://github.com/awslabs/dgl-lifesci/tree/master/examples/link_prediction/ogbl-ppa), [PyTorch on ogbg-ppa](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/ogbg_ppa), [TensorFlow](../examples/tensorflow/gcn), [MXNet](../examples/mxnet/gcn)
    - Tags: node classification, link prediction, graph classification, OGB

- <a name="capsule"></a> Sabour et al. Dynamic Routing Between Capsules. [Paper link](https://arxiv.org/abs/1710.09829).
    - Example code: [PyTorch](../examples/pytorch/capsule)
    - Tags: image classification
  
- <a name="gcmc"></a> van den Berg et al. Graph Convolutional Matrix Completion. [Paper link](https://arxiv.org/abs/1706.02263).
    - Example code: [PyTorch](../examples/pytorch/gcmc)
    - Tags: matrix completion, recommender system, link prediction, bipartite graphs

- <a name="graphsage"></a> Hamilton et al. Inductive Representation Learning on Large Graphs. [Paper link](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf).
    - Example code: [PyTorch](../examples/pytorch/graphsage), [PyTorch on ogbn-products](../examples/pytorch/ogb/ogbn-products), [PyTorch on ogbl-ppa](https://github.com/awslabs/dgl-lifesci/tree/master/examples/link_prediction/ogbl-ppa), [MXNet](../examples/mxnet/graphsage)
    - Tags: node classification, sampling, unsupervised learning, link prediction, OGB

- <a name="metapath2vec"></a> Dong et al. metapath2vec: Scalable Representation Learning for Heterogeneous Networks. [Paper link](https://dl.acm.org/doi/10.1145/3097983.3098036).
    - Example code: [PyTorch](../examples/pytorch/metapath2vec)
    - Tags: heterogeneous graphs, network embedding, large-scale, node classification

- <a name="tagcn"></a> Du et al. Topology Adaptive Graph Convolutional Networks. [Paper link](https://arxiv.org/abs/1710.10370).
    - Example code: [PyTorch](../examples/pytorch/tagcn), [MXNet](../examples/mxnet/tagcn)
    - Tags: node classification
    
- <a name="pointnet"></a> Qi et al. PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. [Paper link](https://arxiv.org/abs/1612.00593).
    - Example code: [PyTorch](../examples/pytorch/pointcloud/pointnet)
    - Tags: point cloud classification, point cloud part-segmentation

- <a name="pointnet++"></a> Qi et al. PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. [Paper link](https://arxiv.org/abs/1706.02413).
    - Example code: [PyTorch](../examples/pytorch/pointcloud/pointnet)
    - Tags: point cloud classification
    
- <a name="rgcn"></a> Schlichtkrull. Modeling Relational Data with Graph Convolutional Networks. [Paper link](https://arxiv.org/abs/1703.06103).
    - Example code: [PyTorch example using homogeneous DGLGraphs](../examples/pytorch/rgcn), [PyTorch](../examples/pytorch/rgcn-hetero), [TensorFlow](../examples/tensorflow/rgcn), [MXNet](../examples/mxnet/rgcn)
    - Tags: node classification, link prediction, heterogeneous graphs, sampling

- <a name="transformer"></a> Vaswani et al. Attention Is All You Need. [Paper link](https://arxiv.org/abs/1706.03762).
    - Example code: [PyTorch](../examples/pytorch/transformer)
    - Tags: machine translation

- <a name="mpnn"></a> Gilmer et al. Neural Message Passing for Quantum Chemistry. [Paper link](https://arxiv.org/abs/1704.01212).
    - Example code: [PyTorch](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/alchemy), [PyTorch for custom data](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/csv_data_configuration)
    - Tags: molecules, quantum chemistry

- <a name="acnn"></a> Gomes et al. Atomic Convolutional Networks for Predicting Protein-Ligand Binding Affinity. [Paper link](https://arxiv.org/abs/1703.10603).
    - Example code: [PyTorch](https://github.com/awslabs/dgl-lifesci/tree/master/examples/binding_affinity_prediction)
    - Tags: binding affinity prediction, molecules, proteins

- <a name="schnet"></a> Schütt et al. SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. [Paper link](https://arxiv.org/abs/1706.08566).
    - Example code: [PyTorch](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/alchemy)
    - Tags: molecules, quantum chemistry

- <a name="dcrnn"></a> Li et al. Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forcasting. [Paper link](https://arxiv.org/abs/1707.01926).
    - Example code: [Pytorch](../examples/pytorch/dtgrnn)
    - Tags: Static discrete temporal graph, traffic forcasting.

## 2016

- <a name="ggnn"></a> Li et al. Gated Graph Sequence Neural Networks. [Paper link](https://arxiv.org/abs/1511.05493).
    - Example code: [PyTorch](../examples/pytorch/ggnn)
    - Tags: question answering
- <a name="chebnet"></a> Defferrard et al. Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering. [Paper link](https://arxiv.org/abs/1606.09375).
    - Example code: [PyTorch on image classification](../examples/pytorch/model_zoo/geometric), [PyTorch on node classification](../examples/pytorch/model_zoo/citation_network)
    - Tags: image classification, graph classification, node classification
- <a name="monet"></a> Monti et al. Geometric deep learning on graphs and manifolds using mixture model CNNs. [Paper link](https://arxiv.org/abs/1611.08402).
    - Example code: [PyTorch on image classification](../examples/pytorch/model_zoo/geometric), [PyTorch on node classification](../examples/pytorch/monet), [MXNet on node classification](../examples/mxnet/monet)
    - Tags: image classification, graph classification, node classification
- <a name="weave"></a> Kearnes et al. Molecular Graph Convolutions: Moving Beyond Fingerprints. [Paper link](https://arxiv.org/abs/1603.00856).
    - Example code: [PyTorch](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/moleculenet), [PyTorch for custom data](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/csv_data_configuration)
    - Tags: molecular property prediction
- <a name="complex"></a> Trouillon et al. Complex Embeddings for Simple Link Prediction. [Paper link](http://proceedings.mlr.press/v48/trouillon16.pdf).
    - Example code: [PyTorch](https://github.com/awslabs/dgl-ke/tree/master/examples), [PyTorch for custom data](https://aws-dglke.readthedocs.io/en/latest/commands.html)
    - Tags: knowledge graph embedding
- <a name="vgae"></a> Thomas et al. Variational Graph Auto-Encoders. [Paper link](https://arxiv.org/abs/1611.07308).
    - Example code: [PyTorch](../examples/pytorch/vgae)
    - Tags: link prediction

## 2015

- <a name="line"></a> Tang et al. LINE: Large-scale Information Network Embedding. [Paper link](https://arxiv.org/abs/1503.03578).
    - Example code: [PyTorch on OGB](../examples/pytorch/ogb/line)
    - Tags: network embedding, transductive learning, OGB, link prediction

- <a name="treelstm"></a> Sheng Tai et al. Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks. [Paper link](https://arxiv.org/abs/1503.00075).
    - Example code: [PyTorch](../examples/pytorch/tree_lstm), [MXNet](../examples/mxnet/tree_lstm)
    - Tags: sentiment classification
    
- <a name="seq2seq"></a> Vinyals et al. Order Matters: Sequence to sequence for sets. [Paper link](https://arxiv.org/abs/1511.06391).
    - Pooling module: [PyTorch](https://docs.dgl.ai/api/python/nn.pytorch.html#set2set), [MXNet](https://docs.dgl.ai/api/python/nn.mxnet.html#set2set)
    - Tags: graph classification
    
- <a name="transr"></a> Lin et al. Learning Entity and Relation Embeddings for Knowledge Graph Completion. [Paper link](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewPaper/9571).
    - Example code: [PyTorch](https://github.com/awslabs/dgl-ke/tree/master/examples), [PyTorch for custom data](https://aws-dglke.readthedocs.io/en/latest/commands.html)
    - Tags: knowledge graph embedding

- <a name="distmul"></a> Yang et al. Embedding Entities and Relations for Learning and Inference in Knowledge Bases. [Paper link](https://arxiv.org/abs/1412.6575).
    - Example code: [PyTorch](https://github.com/awslabs/dgl-ke/tree/master/examples), [PyTorch for custom data](https://aws-dglke.readthedocs.io/en/latest/commands.html)
    - Tags: knowledge graph embedding

- <a name="nf"></a> Duvenaud et al. Convolutional Networks on Graphs for Learning Molecular Fingerprints. [Paper link](https://arxiv.org/abs/1509.09292).
    - Example code: [PyTorch](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/moleculenet), [PyTorch for custom data](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/csv_data_configuration)
    - Tags: molecules, molecular property prediction

## 2014

- <a name="deepwalk"></a> Perozzi et al. DeepWalk: Online Learning of Social Representations. [Paper link](https://arxiv.org/abs/1403.6652).
    - Example code: [PyTorch on OGB](../examples/pytorch/ogb/deepwalk)
    - Tags: network embedding, transductive learning, OGB, link prediction

- <a name="hausdorff"></a> Fischer et al. A Hausdorff Heuristic for Efficient Computation of Graph Edit Distance. [Paper link](https://link.springer.com/chapter/10.1007/978-3-662-44415-3_9).
    - Example code: [PyTorch](../examples/pytorch/graph_matching)
    - Tags: graph edit distance, graph matching

## 2013

- <a name="transe"></a> Bordes et al. Translating Embeddings for Modeling Multi-relational Data. [Paper link](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf).
    - Example code: [PyTorch](https://github.com/awslabs/dgl-ke/tree/master/examples), [PyTorch for custom data](https://aws-dglke.readthedocs.io/en/latest/commands.html)
    - Tags: knowledge graph embedding

## 2011

- <a name="bipartite"></a> Fankhauser et al. Speeding Up Graph Edit Distance Computation through Fast Bipartite Matching. [Paper link](https://link.springer.com/chapter/10.1007/978-3-642-20844-7_11).
    - Example code: [PyTorch](../examples/pytorch/graph_matching)
    - Tags: graph edit distance, graph matching

- <a name="rescal"></a> Nickel et al. A Three-Way Model for Collective Learning on Multi-Relational Data. [Paper link](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.2015&rep=rep1&type=pdf).
    - Example code: [PyTorch](https://github.com/awslabs/dgl-ke/tree/master/examples), [PyTorch for custom data](https://aws-dglke.readthedocs.io/en/latest/commands.html)
    - Tags: knowledge graph embedding

## 2010

- <a name="lda"></a> Hoffman et al. Online Learning for Latent Dirichlet Allocation. [Paper link](https://papers.nips.cc/paper/2010/file/71f6278d140af599e06ad9bf1ba03cb0-Paper.pdf).
    - Example code: [PyTorch](../examples/pytorch/lda)
    - Tags: sklearn, decomposition, latent Dirichlet allocation

## 2009

- <a name="astar"></a> Riesen et al. Speeding Up Graph Edit Distance Computation with a Bipartite Heuristic. [Paper link](https://core.ac.uk/download/pdf/33054885.pdf).
    - Example code: [PyTorch](../examples/pytorch/graph_matching)
    - Tags: graph edit distance, graph matching

## 2006

- <a name="beam"></a> Neuhaus et al. Fast Suboptimal Algorithms for the Computation of Graph Edit Distance. [Paper link](https://link.springer.com/chapter/10.1007/11815921_17).
    - Example code: [PyTorch](../examples/pytorch/graph_matching)
    - Tags: graph edit distance, graph matching

## 1998

- <a name="pagerank"></a> Page et al. The PageRank Citation Ranking: Bringing Order to the Web. [Paper link](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.38.5427).
    - Example code: [PyTorch](../examples/pytorch/pagerank.py)
    - Tags: PageRank
