# Inductive Graph-based Matrix Completion

Paper link: [https://arxiv.org/abs/1904.12058](https://arxiv.org/abs/1904.12058)
Author's code: [https://github.com/muhanzhang/IGMC](https://github.com/muhanzhang/IGMC)

Credit: Junfeng Zhou ([@zhoujf620](https://github.com/zhoujf620))

## Dependencies

* PyTorch 1.2+
* DGL 0.5 (nightly version)

## Data

Supported datasets: ml-100k, ml-1m

## How to run

- ml-100k

```shell
python3 train.py --data_name ml-100k --testing \
                 --batch_size 128 --edge_dropout 0.2 --max_nodes_per_hop 200 --train_epochs 80 \
                 --device 0
```

- ml-1m

```shell
python3 train_multi_gpu.py --data_name ml-1m --testing \
                --batch_size 128 --edge_dropout 0. --max_nodes_per_hop 100 --train_epochs 40 \
                --train_log_interval 100 --valid_log_interval 5 --train_lr_decay_step 20 \
                --gpu 0,1,2,3
```

## Results

|Dataset|Our code <br> best of epochs|Author code <br> best of epochs / ensembled|
|:-:|:-:|:-:|
|ml-100k|0.9041|0.9053 / 0.9051|
|ml-1m|0.8697|0.8685 / 0.8558|
