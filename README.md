<p align="center">
  <img src="http://data.dgl.ai/asset/logo.jpg" height="200">
</p>

[![PyPi Latest Release](https://img.shields.io/pypi/v/dgl.svg)](https://pypi.org/project/dgl/)
[![Conda Latest Release](https://anaconda.org/dglteam/dgl/badges/version.svg)](https://anaconda.org/dglteam/dgl)
[![Build Status](https://ci.dgl.ai/buildStatus/icon?job=DGL/master)](https://ci.dgl.ai/job/DGL/job/master/)
[![Benchmark by ASV](http://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat)](https://asv.dgl.ai/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

Documentation ([Latest](https://docs.dgl.ai/en/latest/) | [Stable](https://docs.dgl.ai)) | [DGL at a glance](https://docs.dgl.ai/tutorials/basics/1_first.html#sphx-glr-tutorials-basics-1-first-py) | [Model Tutorials](https://docs.dgl.ai/tutorials/models/index.html) | [Official Examples](examples/README.md) | [Discussion Forum](https://discuss.dgl.ai) | [Slack Channel](https://join.slack.com/t/deep-graph-library/shared_invite/zt-eb4ict1g-xcg3PhZAFAB8p6dtKuP6xQ)

**For a full list of official DGL examples, see [here](examples).**

DGL is an easy-to-use, high performance and scalable Python package for deep learning on graphs. DGL is framework agnostic, meaning if a deep graph model is a component of an end-to-end application, the rest of the logics can be implemented in any major frameworks, such as PyTorch, Apache MXNet or TensorFlow.

<p align="center">
  <img src="http://data.dgl.ai/asset/image/DGL-Arch.png" alt="DGL v0.4 architecture" width="600">
  <br>
  <b>Figure</b>: DGL Overall Architecture
</p>

## <img src="http://data.dgl.ai/asset/image/new.png" width="30">DGL News
**02/26/2021**: The new **v0.6.0 release** includes distributed heterogeneous graph support, 13 more model examples, a Chinese translation of user guide thank to community support, and a new tutorial.  See our [release note](https://github.com/dmlc/dgl/releases/tag/v0.6.0) for more details.

**09/05/2020**: We invite you to participate in the survey [here](https://forms.gle/Ej3jHCocACmb49Gp8) to make DGL better fit for your needs.  Thanks!

**08/21/2020**: The new **v0.5.0 release** includes distributed GNN training, overhauled documentation and user guide, and several more features.  We have also submitted some models to the [OGB](https://ogb.stanford.edu) leaderboard.  See our [release note](https://github.com/dmlc/dgl/releases/tag/0.5.0) for more details.

## Using DGL

**A data scientist** may want to apply a pre-trained model to your data right away. For this you can use DGL's [Application packages, formally *Model Zoo*](https://github.com/dmlc/dgl/tree/master/apps). Application packages are developed for domain applications, as is the case for [DGL-LifeScience](https://github.com/awslabs/dgl-lifesci). We will soon add model zoo for knowledge graph embedding learning and recommender systems. Here's how you will use a pretrained model:
```python
from dgllife.data import Tox21
from dgllife.model import load_pretrained
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

dataset = Tox21(smiles_to_bigraph, CanonicalAtomFeaturizer())
model = load_pretrained('GCN_Tox21') # Pretrained model loaded
model.eval()

smiles, g, label, mask = dataset[0]
feats = g.ndata.pop('h')
label_pred = model(g, feats)
print(smiles)                   # CCOc1ccc2nc(S(N)(=O)=O)sc2c1
print(label_pred[:, mask != 0]) # Mask non-existing labels
# tensor([[ 1.4190, -0.1820,  1.2974,  1.4416,  0.6914,  
# 2.0957,  0.5919,  0.7715, 1.7273,  0.2070]])
```

**Further reading**: DGL is released as a managed service on AWS SageMaker, see the medium posts for an easy trip to DGL on SageMaker([part1](https://medium.com/@julsimon/a-primer-on-graph-neural-networks-with-amazon-neptune-and-the-deep-graph-library-5ce64984a276) and [part2](https://medium.com/@julsimon/deep-graph-library-part-2-training-on-amazon-sagemaker-54d318dfc814)).

**Researchers** can start from the growing list of [models implemented in DGL](https://github.com/dmlc/dgl/tree/master/examples). Developing new models does not mean that you have to start from scratch. Instead, you can reuse many [pre-built modules](https://docs.dgl.ai/api/python/nn.html). Here is how to get a standard two-layer graph convolutional model with a pre-built GraphConv module:
```python
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F

# build a two-layer GCN with ReLU as the activation in between
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, num_classes)
    
    def forward(self, graph, inputs):
        h = self.gcn_layer1(graph, inputs)
        h = F.relu(h)
        h = self.gcn_layer2(graph, h)
        return h
```

Next level down, you may want to innovate your own module. DGL offers a succinct message-passing interface (see tutorial [here](https://docs.dgl.ai/tutorials/basics/3_pagerank.html)). Here is how Graph Attention Network (GAT) is implemented ([complete codes](https://docs.dgl.ai/api/python/nn.pytorch.html#gatconv)). Of course, you can also find GAT as a module [GATConv](https://docs.dgl.ai/api/python/nn.pytorch.html#gatconv):
```python
import torch.nn as nn
import torch.nn.functional as F

# Define a GAT layer
class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GATLayer, self).__init__()
        self.linear_func = nn.Linear(in_feats, out_feats, bias=False)
        self.attention_func = nn.Linear(2 * out_feats, 1, bias=False)
        
    def edge_attention(self, edges):
        concat_z = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        src_e = self.attention_func(concat_z)
        src_e = F.leaky_relu(src_e)
        return {'e': src_e}
    
    def message_func(self, edges):
        return {'z': edges.src['z'], 'e':edges.data['e']}
        
    def reduce_func(self, nodes):
        a = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(a * nodes.mailbox['z'], dim=1)
        return {'h': h}
                               
    def forward(self, graph, h):
        z = self.linear_func(h)
        graph.ndata['z'] = z
        graph.apply_edges(self.edge_attention)
        graph.update_all(self.message_func, self.reduce_func)
        return graph.ndata.pop('h')
```
## Performance and Scalability

**Microbenchmark on speed and memory usage**: While leaving tensor and autograd functions to backend frameworks (e.g. PyTorch, MXNet, and TensorFlow), DGL aggressively optimizes storage and computation with its own kernels. Here's a comparison to another popular package -- PyTorch Geometric (PyG). The short story is that raw speed is similar, but DGL has much better memory management.


| Dataset  |    Model     |                   Accuracy                   |                    Time <br> PyG &emsp;&emsp; DGL                    |           Memory <br> PyG &emsp;&emsp; DGL            |
| -------- |:------------:|:--------------------------------------------:|:--------------------------------------------------------------------:|:-----------------------------------------------------:|
| Cora     | GCN <br> GAT | 81.31 &plusmn; 0.88 <br> 83.98 &plusmn; 0.52 | <b>0.478</b> &emsp;&emsp; 0.666 <br> 1.608 &emsp;&emsp; <b>1.399</b> | 1.1 &emsp;&emsp; 1.1 <br> 1.2 &emsp;&emsp; <b>1.1</b> |
| CiteSeer | GCN <br> GAT | 70.98 &plusmn; 0.68 <br> 69.96 &plusmn; 0.53 | <b>0.490</b> &emsp;&emsp; 0.674 <br> 1.606 &emsp;&emsp; <b>1.399</b> | 1.1 &emsp;&emsp; 1.1 <br> 1.3 &emsp;&emsp; <b>1.1</b> |
| PubMed   | GCN <br> GAT | 79.00 &plusmn; 0.41 <br> 77.65 &plusmn; 0.32 | <b>0.491</b> &emsp;&emsp; 0.690 <br> 1.946 &emsp;&emsp; <b>1.393</b> | 1.1 &emsp;&emsp; 1.1 <br> 1.6 &emsp;&emsp; <b>1.1</b> |
| Reddit   |     GCN      |             93.46 &plusmn; 0.06              |                    *OOM*&emsp;&emsp; <b>28.6</b>                     |            *OOM* &emsp;&emsp; <b>11.7</b>             |
| Reddit-S |     GCN      |                     N/A                      |                    29.12 &emsp;&emsp; <b>9.44</b>                    |             15.7 &emsp;&emsp; <b>3.6</b>              |

Table: Training time(in seconds) for 200 epochs and memory consumption(GB)

Here is another comparison of DGL on TensorFlow backend with other TF-based GNN tools (training time in seconds for one epoch):

| Dateset | Model | DGL | GraphNet | tf_geometric |
| ------- | ----- | --- | -------- | ------------ |
| Core | GCN | 0.0148 | 0.0152 | 0.0192 |
| Reddit | GCN | 0.1095 | OOM | OOM |
| PubMed | GCN | 0.0156 | 0.0553 | 0.0185 |
| PPI | GCN | 0.09 | 0.16 | 0.21 |
| Cora | GAT | 0.0442 | n/a | 0.058 |
| PPI | GAT | 0.398 | n/a | 0.752 |

High memory utilization allows DGL to push the limit of single-GPU performance, as seen in below images.
| <img src="http://data.dgl.ai/asset/image/DGLvsPyG-time1.png" width="400"> | <img src="http://data.dgl.ai/asset/image/DGLvsPyG-time2.png" width="400"> |
| -------- | -------- |

**Scalability**: DGL has fully leveraged multiple GPUs in both one machine and clusters for increasing training speed, and has better performance than alternatives, as seen in below images.

<p align="center">
  <img src="http://data.dgl.ai/asset/image/one-four-GPUs.png" width="600">
</p>

| <img src="http://data.dgl.ai/asset/image/one-four-GPUs-DGLvsGraphVite.png"> |  <img src="http://data.dgl.ai/asset/image/one-fourMachines.png"> | 
| :---------------------------------------: | -- |


**Further reading**: Detailed comparison of DGL and other Graph alternatives can be found [here](https://arxiv.org/abs/1909.01315).

## DGL Models and Applications

### DGL for research
Overall there are 30+ models implemented by using DGL:
- [PyTorch](https://github.com/dmlc/dgl/tree/master/examples/pytorch)
- [MXNet](https://github.com/dmlc/dgl/tree/master/examples/mxnet)
- [TensorFlow](https://github.com/dmlc/dgl/tree/master/examples/tensorflow)

### DGL for domain applications
- [DGL-LifeSci](https://github.com/awslabs/dgl-lifesci), previously DGL-Chem
- [DGL-KE](https://github.com/awslabs/dgl-ke)
- DGL-RecSys(coming soon)

### DGL for NLP/CV problems
- [TreeLSTM](https://github.com/dmlc/dgl/tree/master/examples/pytorch/tree_lstm)
- [GraphWriter](https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphwriter)
- [Capsule Network](https://github.com/dmlc/dgl/tree/master/examples/pytorch/capsule)

We are currently in Beta stage.  More features and improvements are coming.

## Awesome Papers Using DGL

1. [**Benchmarking Graph Neural Networks**](https://arxiv.org/pdf/2003.00982.pdf), *Vijay Prakash Dwivedi, Chaitanya K. Joshi, Thomas Laurent, Yoshua Bengio, Xavier Bresson*

1. [**Open Graph Benchmarks: Datasets for Machine Learning on Graphs**](https://arxiv.org/pdf/2005.00687.pdf), NeurIPS'20, *Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, Jure Leskovec*

1. [**DropEdge: Towards Deep Graph Convolutional Networks on Node Classification**](https://openreview.net/pdf?id=Hkx1qkrKPr), ICLR'20, *Yu Rong, Wenbing Huang, Tingyang Xu, Junzhou Huan*

1. [**Discourse-Aware Neural Extractive Text Summarization**](https://www.aclweb.org/anthology/2020.acl-main.451/), ACL'20, *Jiacheng Xu, Zhe Gan, Yu Cheng, Jingjing Liu*

1. [**GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training**](https://dl.acm.org/doi/pdf/10.1145/3394486.3403168?casa_token=EClsH2Vc4DcAAAAA:LIB8cbtr6yTDbYuv4cTLwTIYeDq5Y2dhj_ktcWdKpzdPLGeiuL0o8GlcN4QIOnpsAnmGeGVZ), KDD'20, *Jiezhong Qiu, Qibin Chen, Yuxiao Dong, Jing Zhang, Hongxia Yang, Ming Ding, Kuansan Wang, Jie Tang*

1. [**DGL-KE: Training Knowledge Graph Embeddings at Scale**](https://arxiv.org/pdf/2004.08532), SIGIR'20, *Da Zheng, Xiang Song, Chao Ma, Zeyuan Tan, Zihao Ye, Jin Dong, Hao Xiong, Zheng Zhang, George Karypis*

1. [**Improving Graph Neural Network Expressivity via Subgraph Isomorphism Counting**](https://arxiv.org/pdf/2006.09252.pdf), *Giorgos Bouritsas, Fabrizio Frasca, Stefanos Zafeiriou, Michael M. Bronstein*

1. [**INT: An Inequality Benchmark for Evaluating Generalization in Theorem Proving**](https://arxiv.org/pdf/2007.02924.pdf), *Yuhuai Wu, Albert Q. Jiang, Jimmy Ba, Roger Grosse*

1. [**Finding Patient Zero: Learning Contagion Source with Graph Neural Networks**](https://arxiv.org/pdf/2006.11913.pdf), *Chintan Shah, Nima Dehmamy, Nicola Perra, Matteo Chinazzi, Albert-László Barabási, Alessandro Vespignani, Rose Yu*

1. [**FeatGraph: A Flexible and Efficient Backend for Graph Neural Network Systems**](https://arxiv.org/pdf/2008.11359.pdf), SC'20, *Yuwei Hu, Zihao Ye, Minjie Wang, Jiali Yu, Da Zheng, Mu Li, Zheng Zhang, Zhiru Zhang, Yida Wang*


<details><summary>more</summary>

11. [**BP-Transformer: Modelling Long-Range Context via Binary Partitioning.**](https://arxiv.org/pdf/1911.04070.pdf), *Zihao Ye, Qipeng Guo, Quan Gan, Xipeng Qiu, Zheng Zhang*

12. [**OptiMol: Optimization of Binding Affinities in Chemical Space for Drug Discovery**](https://www.biorxiv.org/content/biorxiv/early/2020/06/16/2020.05.23.112201.full.pdf), *Jacques Boitreaud,Vincent Mallet, Carlos Oliver, Jérôme Waldispühl*

1. [**JAKET: Joint Pre-training of Knowledge Graph and Language Understanding**](https://arxiv.org/pdf/2010.00796.pdf), *Donghan Yu, Chenguang Zhu, Yiming Yang, Michael Zeng*

1. [**Architectural Implications of Graph Neural Networks**](https://arxiv.org/pdf/2009.00804.pdf), *Zhihui Zhang, Jingwen Leng, Lingxiao Ma, Youshan Miao, Chao Li, Minyi Guo*

1. [**Combining Reinforcement Learning and Constraint Programming for Combinatorial Optimization**](https://arxiv.org/pdf/2006.01610.pdf), *Quentin Cappart, Thierry Moisan, Louis-Martin Rousseau1, Isabeau Prémont-Schwarz, and Andre Cire*

1. [**Therapeutics Data Commons: Machine Learning Datasets and Tasks for Therapeutics**](https://arxiv.org/abs/2102.09548) ([code repo](https://github.com/mims-harvard/TDC)), *Kexin Huang, Tianfan Fu, Wenhao Gao, Yue Zhao, Yusuf Roohani, Jure Leskovec, Connor W. Coley, Cao Xiao, Jimeng Sun, Marinka Zitnik*

1. [**Sparse Graph Attention Networks**](https://arxiv.org/abs/1912.00552), *Yang Ye, Shihao Ji*

1. [**On Self-Distilling Graph Neural Network**](https://arxiv.org/pdf/2011.02255.pdf), *Yuzhao Chen, Yatao Bian, Xi Xiao, Yu Rong, Tingyang Xu, Junzhou Huang*

1. [**Learning Robust Node Representations on Graphs**](https://arxiv.org/pdf/2008.11416.pdf), *Xu Chen, Ya Zhang, Ivor Tsang, and Yuangang Pan*

1. [**Recurrent Event Network: Autoregressive Structure Inference over Temporal Knowledge Graphs**](https://arxiv.org/abs/1904.05530), *Woojeong Jin, Meng Qu, Xisen Jin, Xiang Ren*

1. [**Graph Neural Ordinary Differential Equations**](https://arxiv.org/abs/1911.07532), *Michael Poli, Stefano Massaroli, Junyoung Park, Atsushi Yamashita, Hajime Asama, Jinkyoo Park*

1. [**FusedMM: A Unified SDDMM-SpMM Kernel for Graph Embedding and Graph Neural Networks**](https://arxiv.org/pdf/2011.06391.pdf), *Md. Khaledur Rahman, Majedul Haque Sujon, , Ariful Azad*

1. [**An Efficient Neighborhood-based Interaction Model for Recommendation on Heterogeneous Graph**](https://arxiv.org/pdf/2007.00216.pdf), KDD'20 *Jiarui Jin, Jiarui Qin, Yuchen Fang, Kounianhua Du, Weinan Zhang, Yong Yu, Zheng Zhang, Alexander J. Smola*

1. [**Learning Interaction Models of Structured Neighborhood on Heterogeneous Information Network**](https://arxiv.org/pdf/2011.12683.pdf), *Jiarui Jin, Kounianhua Du, Weinan Zhang, Jiarui Qin, Yuchen Fang, Yong Yu, Zheng Zhang, Alexander J. Smola*

1. [**Graphein - a Python Library for Geometric Deep Learning and Network Analysis on Protein Structures**](https://www.biorxiv.org/content/10.1101/2020.07.15.204701v1), *Arian R. Jamasb, Pietro Lió, Tom L. Blundell*

1. [**Graph Policy Gradients for Large Scale Robot Control**](https://arxiv.org/abs/1907.03822), *Arbaaz Khan, Ekaterina Tolstaya, Alejandro Ribeiro, Vijay Kumar*

1. [**Heterogeneous Molecular Graph Neural Networks for Predicting Molecule Properties**](https://arxiv.org/abs/2009.12710), *Zeren Shui, George Karypis*

1. [**Could Graph Neural Networks Learn Better Molecular Representation for Drug Discovery? A Comparison Study of Descriptor-based and Graph-based Models**](https://assets.researchsquare.com/files/rs-81439/v1_stamped.pdf), *Dejun Jiang, Zhenxing Wu, Chang-Yu Hsieh, Guangyong Chen, Ben Liao, Zhe Wang, Chao Shen, Dongsheng Cao, Jian Wu, Tingjun Hou*

1. [**Principal Neighbourhood Aggregation for Graph Nets**](https://arxiv.org/abs/2004.05718), *Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Liò, Petar Veličković*

1. [**Collective Multi-type Entity Alignment Between Knowledge Graphs**](https://dl.acm.org/doi/abs/10.1145/3366423.3380289), *Qi Zhu, Hao Wei, Bunyamin Sisman, Da Zheng, Christos Faloutsos, Xin Luna Dong, Jiawei Han*

1. [**Graph Representation Forecasting of Patient's Medical Conditions: towards A Digital Twin**](https://arxiv.org/abs/2009.08299), *Pietro Barbiero, Ramon Viñas Torné, Pietro Lió*

1. [**Relational Graph Learning on Visual and Kinematics Embeddings for Accurate Gesture Recognition in Robotic Surgery**](https://arxiv.org/abs/2011.01619), *Yong-Hao Long, Jie-Ying Wu, Bo Lu, Yue-Ming Jin, Mathias Unberath, Yun-Hui Liu, Pheng-Ann Heng and Qi Dou*

1. [**Dark Reciprocal-Rank: Boosting Graph-Convolutional Self-Localization Network via Teacher-to-student Knowledge Transfer**](https://arxiv.org/abs/2011.00402), *Takeda Koji, Tanaka Kanji*

1. [**Graph InfoClust: Leveraging Cluster-Level Node Information For Unsupervised Graph Representation Learning**](https://arxiv.org/abs/2009.06946), *Costas Mavromatis, George Karypis*

1. [**GraphSeam: Supervised Graph Learning Framework for Semantic UV Mapping**](https://arxiv.org/abs/2011.13748), *Fatemeh Teimury, Bruno Roy, Juan Sebastian Casallas, David macdonald, Mark Coates*

1. [**Comprehensive Study on Molecular Supervised Learning with Graph Neural Networks**](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00416), *Doyeong Hwang, Soojung Yang, Yongchan Kwon, Kyung Hoon Lee, Grace Lee, Hanseok Jo, Seyeol Yoon, and Seongok Ryu*

1. [**A graph auto-encoder model for miRNA-disease associations prediction**](https://academic.oup.com/bib/advance-article-abstract/doi/10.1093/bib/bbaa240/5929824?redirectedFrom=fulltext), *Zhengwei Li, Jiashu Li, Ru Nie, Zhu-Hong You, Wenzheng Bao*

1. [**Graph convolutional regression of cardiac depolarization from sparse endocardial maps**](https://arxiv.org/abs/2009.14068), STACOM 2020 workshop, *Felix Meister, Tiziano Passerini, Chloé Audigier, Èric Lluch, Viorel Mihalef, Hiroshi Ashikaga, Andreas Maier, Henry Halperin, Tommaso Mansi*

1. [**AttnIO: Knowledge Graph Exploration with In-and-Out Attention Flow for Knowledge-Grounded Dialogue**](https://www.aclweb.org/anthology/2020.emnlp-main.280/), EMNLP'20, *Jaehun Jung, Bokyung Son, Sungwon Lyu*

1. [**Learning from Non-Binary Constituency Trees via Tensor Decomposition**](https://github.com/danielecastellana22/tensor-tree-nn), COLING'20, *Daniele Castellana, Davide Bacciu*

1. [**Inducing Alignment Structure with Gated Graph Attention Networks for Sentence Matching**](https://arxiv.org/abs/2010.07668), *Peng Cui, Le Hu, Yuanchao Liu*

1. [**Enhancing Extractive Text Summarization with Topic-Aware Graph Neural Networks**](https://arxiv.org/abs/2010.06253), COLING'20, *Peng Cui, Le Hu, Yuanchao Liu*

1. [**Double Graph Based Reasoning for Document-level Relation Extraction**](https://arxiv.org/abs/2009.13752), EMNLP'20, *Shuang Zeng, Runxin Xu, Baobao Chang, Lei Li*

1. [**Systematic Generalization on gSCAN with Language Conditioned Embedding**](https://arxiv.org/abs/2009.05552), AACL-IJCNLP'20, *Tong Gao, Qi Huang, Raymond J. Mooney*

1. [**Automatic selection of clustering algorithms using supervised graph embedding**](https://arxiv.org/pdf/2011.08225.pdf), *Noy Cohen-Shapira, Lior Rokach*

1. [**Improving Learning to Branch via Reinforcement Learning**](https://openreview.net/forum?id=z4D7-PTxTb), *Haoran Sun, Wenbo Chen, Hui Li, Le Song*

1. [**A Practical Guide to Graph Neural Networks**](https://arxiv.org/pdf/2010.05234.pdf), *Isaac Ronald Ward, Jack Joyner, Casey Lickfold, Stash Rowe, Yulan Guo, Mohammed Bennamoun*, [code](https://github.com/isolabs/gnn-tutorial)

1. [**APAN: Asynchronous Propagation Attention Network for Real-time Temporal Graph Embedding**](https://arxiv.org/pdf/2011.11545.pdf), SIGMOD'21, *Xuhong Wang, Ding Lyu, Mengjian Li, Yang Xia, Qi Yang, Xinwen Wang, Xinguang Wang, Ping Cui, Yupu Yang, Bowen Sun, Zhenyu Guo, Junkui Li*

1. [**Uncertainty-Matching Graph Neural Networks to Defend Against Poisoning Attacks**](https://arxiv.org/pdf/2009.14455.pdf), *Uday Shankar Shanthamallu, Jayaraman J. Thiagarajan, Andreas Spanias*

1. [**Computing Graph Neural Networks: A Survey from Algorithms to Accelerators**](https://arxiv.org/pdf/2010.00130.pdf), *Sergi Abadal, Akshay Jain, Robert Guirado, Jorge López-Alonso, Eduard Alarcón*

1. [**NHK_STRL at WNUT-2020 Task 2: GATs with Syntactic Dependencies as Edges and CTC-based Loss for Text Classification**](https://www.aclweb.org/anthology/2020.wnut-1.43.pdf), *Yuki Yasuda, Taichi Ishiwatari, Taro Miyazaki, Jun Goto*

1. [**Relation-aware Graph Attention Networks with Relational Position Encodings for Emotion Recognition in Conversations**](https://www.aclweb.org/anthology/2020.emnlp-main.597.pdf), *Taichi Ishiwatari, Yuki Yasuda, Taro Miyazaki, Jun Goto*

1. [**PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks**](https://proceedings.neurips.cc/paper/2020/file/8fb134f258b1f7865a6ab2d935a897c9-Paper.pdf), *Minh N. Vu, My T. Thai*

1. [**A Generalization of Transformer Networks to Graphs**](https://arxiv.org/pdf/2012.09699.pdf), *Vijay Prakash Dwivedi, Xavier Bresson* 

1. [**Discourse-Aware Neural Extractive Text Summarization**](https://www.aclweb.org/anthology/2020.acl-main.451.pdf), ACL'20, *Jiacheng Xu, Zhe Gan, Yu Cheng, Jingjing Liu*

1. [**Learning Robust Node Representations on Graphs**](https://arxiv.org/abs/2008.11416), *Xu Chen, Ya Zhang, Ivor Tsang, Yuangang Pan*

1. [**Adaptive Graph Diffusion Networks with Hop-wise Attention**](https://arxiv.org/abs/2012.15024), *Chuxiong Sun, Guoshi Wu*

1. [**The Photoswitch Dataset: A Molecular Machine Learning Benchmark for the Advancement of Synthetic Chemistry**](https://arxiv.org/abs/2008.03226), *Aditya R. Thawani, Ryan-Rhys Griffiths, Arian Jamasb, Anthony Bourached, Penelope Jones, William McCorkindale, Alexander A. Aldrick, Alpha A. Lee*

1. [**A community-powered search of machine learning strategy space to find NMR property prediction models**](https://arxiv.org/abs/2008.05994), *Lars A. Bratholm, Will Gerrard, Brandon Anderson, Shaojie Bai, Sunghwan Choi, Lam Dang, Pavel Hanchar, Addison Howard, Guillaume Huard, Sanghoon Kim, Zico Kolter, Risi Kondor, Mordechai Kornbluth, Youhan Lee, Youngsoo Lee, Jonathan P. Mailoa, Thanh Tu Nguyen, Milos Popovic, Goran Rakocevic, Walter Reade, Wonho Song, Luka Stojanovic, Erik H. Thiede, Nebojsa Tijanic, Andres Torrubia, Devin Willmott, Craig P. Butts, David R. Glowacki, Kaggle participants*

1. [**Adaptive Layout Decomposition with Graph Embedding Neural Networks**](http://www.cse.cuhk.edu.hk/~byu/papers/C98-DAC2020-MPL-Selector.pdf), *Wei Li, Jialu Xia, Yuzhe Ma, Jialu Li, Yibo Lin, Bei Yu*, DAC'20

1. [**Transfer Learning with Graph Neural Networks for Optoelectronic Properties of Conjugated Oligomers**](https://aip.scitation.org/doi/10.1063/5.0037863), J. Chem. Phys. 154, *Chee-Kong Lee, Chengqiang Lu, Yue Yu, Qiming Sun, Chang-Yu Hsieh, Shengyu Zhang, Qi Liu, and  Liang Shi*

1. [**Jet tagging in the Lund plane with graph networks**](https://link.springer.com/article/10.1007/JHEP03(2021)052), Journal of High Energy Physics 2021, *Frédéric A. Dreyer and Huilin Qu* 

1. [**Global Attention Improves Graph Networks Generalization**](https://arxiv.org/abs/2006.07846), *Omri Puny, Heli Ben-Hamu, and Yaron Lipman*

1. [**Learning over Families of Sets -- Hypergraph Representation Learning for Higher Order Tasks**](https://arxiv.org/abs/2101.07773), SDM 2021, *Balasubramaniam Srinivasan, Da Zheng, and George Karypis*

1. [**SSFG: Stochastically Scaling Features and Gradients for Regularizing Graph Convolution Networks**](https://arxiv.org/abs/2102.10338), *Haimin Zhang, Min Xu*

1. [**Application and evaluation of knowledge graph embeddings in biomedical data**](https://peerj.com/articles/cs-341/), PeerJ Computer Science 7:e341, *Mona Alshahrani​, Maha A. Thafar, Magbubah Essack*

1. [**MoTSE: an interpretable task similarity estimator for small molecular property prediction tasks**](https://www.biorxiv.org/content/10.1101/2021.01.13.426608v2), bioRxiv 2021.01.13.426608, *Han Li, Xinyi Zhao, Shuya Li, Fangping Wan, Dan Zhao, Jianyang Zeng*

1. [**Reinforcement Learning For Data Poisoning on Graph Neural Networks**](https://arxiv.org/abs/2102.06800), *Jacob Dineen, A S M Ahsan-Ul Haque, Matthew Bielskas*

1. [**Generalising Recursive Neural Models by Tensor Decomposition**](https://github.com/danielecastellana22/tensor-tree-nn), IJCNN'20, *Daniele Castellana, Davide Bacciu*

1. [**Tensor Decompositions in Recursive Neural Networks for Tree-Structured Data**](https://github.com/danielecastellana22/tensor-tree-nn), ESANN'20, *Daniele Castellana, Davide Bacciu*

1. [**Combining Self-Organizing and Graph Neural Networks for Modeling Deformable Objects in Robotic Manipulation**](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7806087/), Frotiers in Robotics and AI, *Valencia, Angel J., and Pierre Payeur*

1. [**Joint stroke classification and text line grouping in online handwritten documents with edge pooling attention networks**](https://www.sciencedirect.com/science/article/abs/pii/S0031320321000467), Pattern Recognition, *Jun-Yu Ye, Yan-Ming Zhang, Qing Yang, Cheng-Lin Liu*

1. [**Toward Accurate Predictions of Atomic Properties via Quantum Mechanics Descriptors Augmented Graph Convolutional Neural Network: Application of This Novel Approach in NMR Chemical Shifts Predictions**](https://pubs.acs.org/doi/full/10.1021/acs.jpclett.0c02654), The Journal of Physical Chemistry Letters, *Peng Gao, Jie Zhang, Yuzhu Sun, and Jianguo Yu*

1. [**A Graph Neural Network to Model User Comfort in Robot Navigation**](https://arxiv.org/abs/2102.08863), *Pilar Bachiller, Daniel Rodriguez-Criado, Ronit R. Jorvekar, Pablo Bustos, Diego R. Faria, Luis J. Manso*

</details>

## Installation

DGL should work on

* all Linux distributions no earlier than Ubuntu 16.04
* macOS X
* Windows 10

DGL requires Python 3.6 or later.

Right now, DGL works on [PyTorch](https://pytorch.org) 1.5.0+, [MXNet](https://mxnet.apache.org) 1.6+, and [TensorFlow](https://tensorflow.org) 2.3+.


### Using anaconda

```
conda install -c dglteam dgl           # cpu version
conda install -c dglteam dgl-cuda9.2   # CUDA 9.2
conda install -c dglteam dgl-cuda10.1  # CUDA 10.1
conda install -c dglteam dgl-cuda10.2  # CUDA 10.2
conda install -c dglteam dgl-cuda11.0  # CUDA 11.0
conda install -c dglteam dgl-cuda11.1  # CUDA 11.1
```

### Using pip


|           | Latest Nightly Build Version  | Stable Version          |
|-----------|-------------------------------|-------------------------|
| CPU       | `pip install --pre dgl`       | `pip install dgl`       |
| CUDA 9.2  | `pip install --pre dgl-cu92`  | `pip install dgl-cu92`  |
| CUDA 10.1 | `pip install --pre dgl-cu101` | `pip install dgl-cu101` |
| CUDA 10.2 | `pip install --pre dgl-cu102` | `pip install dgl-cu102` |
| CUDA 11.0 | `pip install --pre dgl-cu110` | `pip install dgl-cu110` |
| CUDA 11.1 | `pip install --pre dgl-cu111` | `pip install dgl-cu111` |

### Built from source code

Refer to the guide [here](https://docs.dgl.ai/install/index.html#install-from-source).


## DGL Major Releases

| Releases  | Date   | Features |
|-----------|--------|-------------------------|
| v0.4.3    | 03/31/2020 | - TensorFlow support <br> - DGL-KE <br> - DGL-LifeSci <br> - Heterograph sampling APIs (experimental) |
| v0.4.2      | 01/24/2020 |  - Heterograph support <br> - TensorFlow support (experimental) <br> - MXNet GNN modules <br> | 
| v0.3.1 | 08/23/2019 | - APIs for GNN modules <br> - Model zoo (DGL-Chem) <br> - New installation |
| v0.2 | 03/09/2019 | - Graph sampling APIs <br> - Speed improvement |
| v0.1 | 12/07/2018 | - Basic DGL APIs <br> - PyTorch and MXNet support <br> - GNN model examples and tutorials |

## New to Deep Learning and Graph Deep Learning?

Check out the open source book [*Dive into Deep Learning*](https://d2l.ai/).

For those who are new to graph neural network, please see the [basic of DGL](https://docs.dgl.ai/tutorials/basics/index.html).

For audience who are looking for more advanced, realistic, and end-to-end examples, please see [model tutorials](https://docs.dgl.ai/tutorials/models/index.html).


## Contributing

Please let us know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/dmlc/dgl/issues).

We welcome all contributions from bug fixes to new features and extensions.

We expect all contributions discussed in the issue tracker and going through PRs.  Please refer to our [contribution guide](https://docs.dgl.ai/contribute.html).

## Cite

If you use DGL in a scientific publication, we would appreciate citations to the following paper:
```
@article{wang2019dgl,
    title={Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks},
    author={Minjie Wang and Da Zheng and Zihao Ye and Quan Gan and Mufei Li and Xiang Song and Jinjing Zhou and Chao Ma and Lingfan Yu and Yu Gai and Tianjun Xiao and Tong He and George Karypis and Jinyang Li and Zheng Zhang},
    year={2019},
    journal={arXiv preprint arXiv:1909.01315}
}
```

## The Team

DGL is developed and maintained by [NYU, NYU Shanghai, AWS Shanghai AI Lab, and AWS MXNet Science Team](https://www.dgl.ai/pages/about.html).

## License

DGL uses Apache License 2.0.
