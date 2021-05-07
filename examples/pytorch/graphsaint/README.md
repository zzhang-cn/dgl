# GraphSAINT

This DGL example implements the paper: GraphSAINT: Graph Sampling Based Inductive Learning Method.

Paper link: https://arxiv.org/abs/1907.04931

Author's code: https://github.com/GraphSAINT/GraphSAINT

Contributor: Liu Tang ([@lt610](https://github.com/lt610))

## Dependencies

- Python 3.7.0
- PyTorch 1.6.0
- NumPy 1.19.2
- Scikit-learn 0.23.2
- DGL 0.5.3

## Dataset

All datasets used are provided by Author's [code](https://github.com/GraphSAINT/GraphSAINT). They are available in [Google Drive](https://drive.google.com/drive/folders/1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) (alternatively, [Baidu Wangpan (code: f1ao)](https://pan.baidu.com/s/1SOb0SiSAXavwAcNqkttwcg#list/path=%2F)). Dataset summary("m" stands for multi-label classification, and "s" for single-label.):
| Dataset | Nodes | Edges | Degree | Feature | Classes | Train/Val/Test |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PPI | 14,755 | 225,270 | 15 | 50 | 121(m) | 0.66/0.12/0.22 |
| Flickr | 89,250 | 899,756 | 10 | 500 | 7(s) | 0.50/0.25/0.25 |
| Reddit | 232,965 | 11,606,919 | 50 | 602 | 41(s) | 0.66/0.10/0.24 |
| Yelp | 716,847 | 6,877,410 | 10 | 300 | 100(m) | 0.75/0.10/0.15 |
| Amazon | 1,598,960 | 132,169,734 | 83 | 200 | 107(m) | 0.85/0.05/0.10 |

## Minibatch training

Run with following:
```bash
python train_sampling.py --gpu 0 --dataset ppi --sampler node --node-budget 6000 --num-repeat 50 --n-epochs 1000 --n-hidden 512 --arch 1-0-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset ppi --sampler edge --edge-budget 4000 --num-repeat 50 --n-epochs 1000 --n-hidden 512 --arch 1-0-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset ppi --sampler rw --num-roots 3000 --length 2 --num-repeat 50 --n-epochs 1000 --n-hidden 512 --arch 1-0-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset flickr --sampler node --node-budget 8000 --num-repeat 25 --n-epochs 30 --n-hidden 256 --arch 1-1-0 --batch-norm --dropout 0.2
python train_sampling.py --gpu 0 --dataset flickr --sampler edge --edge-budget 6000 --num-repeat 25 --n-epochs 15 --n-hidden 256 --arch 1-1-0 --batch-norm --dropout 0.2
python train_sampling.py --gpu 0 --dataset flickr --sampler rw --num-roots 6000 --length 2 --num-repeat 25 --n-epochs 15 --n-hidden 256 --arch 1-1-0 --batch-norm --dropout 0.2
python train_sampling.py --gpu 0 --dataset reddit --sampler node --node-budget 8000 --num-repeat 50 --n-epochs 40 --n-hidden 128 --arch 1-0-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset reddit --sampler edge --edge-budget 6000 --num-repeat 50 --n-epochs 40 --n-hidden 128 --arch 1-0-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset reddit --sampler rw --num-roots 2000 --length 4 --num-repeat 50 --n-epochs 30 --n-hidden 128 --arch 1-0-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset yelp --sampler node --node-budget 5000 --num-repeat 50 --n-epochs 50 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset yelp --sampler edge --edge-budget 2500 --num-repeat 50 --n-epochs 100 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset yelp --sampler rw --num-roots 1250 --length 2 --num-repeat 50 --n-epochs 75 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset amazon --sampler node --node-budget 4500 --num-repeat 50 --n-epochs 30 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset amazon --sampler edge --edge-budget 2000 --num-repeat 50 --n-epochs 30 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset amazon --sampler rw --num-roots 1500 --length 2 --num-repeat 50 --n-epochs 30 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
```

## Comparison

* Paper: results from the paper
* Running: results from experiments with the authors' code
* DGL: results from experiments with the DGL example

### F1-micro

#### Random node sampler

| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Paper | 0.960±0.001 | 0.507±0.001 | 0.962±0.001 | 0.641±0.000 | 0.782±0.004 |
| Running | 0.9628 | 0.5077 | 0.9622 | 0.6393 | 0.7695 |
| DGL | 0.9602 | 0.5094 | 0.9613 | 0.6412 | 0.7702 |

#### Random edge sampler

| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Paper | 0.981±0.007 | 0.510±0.002 | 0.966±0.001 | 0.653±0.003 | 0.807±0.001 |
| Running | 0.9810 | 0.5066 | 0.9656 | 0.6531 | 0.8071 |
| DGL | 0.9811 | 0.5075 | 0.9659 | 0.6530 | exceed |

#### Random walk sampler
| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Paper | 0.981±0.004 | 0.511±0.001 | 0.966±0.001 | 0.653±0.003 | 0.815±0.001 |
| Running | 0.9812 | 0.5104 | 0.9648 | 0.6527 | 0.8131 |
| DGL | 0.9806 | 0.5122 | 0.9662 | 0.6530 | 0.8143 |

### Sampling time

#### Random node sampler

| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Sampling(Running) | 0.77 | 0.65 | 7.46 | 26.29 | 571.42 |
| Sampling(DGL) | 0.24 | 0.57 | 5.06 | 30.04 | 163.75 |
| Normalization(Running) | 0.69 | 2.84 | 11.54 | 32.72 | 407.20 |
| Normalization(DGL) | 1.04 | 0.41 | 21.05 | 68.63 | 2006.94 |

#### Random edge sampler

| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Sampling(Running) | 0.72 | 0.56 | 4.46 | 12.38 | 101.76 |
| Sampling(DGL) | 0.50 | 0.72 | 53.88 | 254.63 | exceed |
| Normalization(Running) | 0.68 | 2.62 | 9.42 | 26.64 | 62.59 |
| Normalization(DGL) | 0.61 | 0.38 | 14.69 | 23.63 | exceed |

#### Random walk sampler

| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Sampling(Running) | 0.83 | 1.22 | 6.69 | 18.84 | 209.83 |
| Sampling(DGL) | 0.28 | 0.63 | 4.02 | 22.01 | 55.09 |
| Normalization(Running) | 0.87 | 2.60 | 10.28 | 24.41 | 145.85 |
| Normalization(DGL) | 0.70 | 0.42 | 18.34 | 32.16 | 683.96 |