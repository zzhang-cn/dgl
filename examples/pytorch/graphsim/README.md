# GraphParticleSim
## DGL Implementation of Interaction-Network paper.

This DGL example implements the GNN model proposed in the paper [Interaction Network](https://arxiv.org/abs/1612.00222.pdf). 

GraphParticleSim implementor
----------------------
This example was implemented by [Ericcsr](https://github.com/Ericcsr) during his Internship work at the AWS Shanghai AI Lab.

The graph dataset used in this example 
---------------------------------------
This Example uses Datasets Generate By Physics N-Body Simulator adapted from [This Repo](https://github.com/jsikyoon/Interaction-networks_tensorflow)

n_body:
    - n Particles/Nodes
    - Complete Bidirectional Graph
    - 10 trajectories should be generated
    - 1000 steps of simulation per trajectory

Dependency
--------------------------------
- ffmpeg 4.3.8
- opencv-python 4.2.0

How to run example files
--------------------------------
In the graphsim folder, run
**Please first run `n_body_sim.py` to generate some data**

Using Ground Truth Velocity From Simulator Directly.

```python
python n_body_sim.py
```

Generate Longer trajectory or more trajectories.

```python
python n_body_sim.py --num_traj <num_traj> --steps <num_steps>
```

**Please use `train.py`**


```python
python train.py --number_workers 15
```

Training with GPU
```python
python train.py --gpu 0 --number_workers 15
```

Training with visualization: for valid visualization, it might take full 40000 epoch of training
```python
python train.py --gpu 0 --number_workers 15 --visualize
```

One Step Loss Performance, Loss of test data after 40000 training epochs.
-------------------------
| Models/Dataset | 6 Body |
| :-------------- | -----: |
| Interaction Network in DGL | 80(10) |
| Interaction Network in Tensorflow | 60 |

-------------------------
Notice that The datasets are generated directly from simulator to prevent using Tensorflow to handle the original dataset. The training is very unstable, the even if the minimum loss is achieved from time to time, there are chances that loss will suddenly increase,in both auther's model and our model. Since the original model hasn't been released, the implementation of this model refers to Tensorflow version implemented in: https://github.com/jsikyoon/Interaction-networks_tensorflow which had consulted the first author for some implementation details.

