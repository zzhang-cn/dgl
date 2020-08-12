## Distributed training

This is an example of training RGCN node classification in a distributed fashion. Currently, the example only support training RGCN graphs with no input features. The current implementation follows ../rgcn/entity_claasify_mp.py.

To train RGCN, it has four steps:

### Step 1: partition the graph.

The example provides a script to partition some builtin graphs such as ogbn-mag graph.
If we want to train RGCN on 4 machines, we need to partition the graph into 4 parts.

In this example, we partition the ogbn-mag graph into 4 parts with Metis. The partitions are balanced with respect to
the number of nodes, the number of edges and the number of labelled nodes.
```bash
python3 partition_graph.py --dataset ogbn-mag --num_parts 4 --balance_train --balance_edges
```

### Step 2: copy the partitioned data to the cluster
DGL provides a script for copying partitioned data to the cluster. The command below copies partition data
to the machines in the cluster. The configuration of the cluster is defined by `ip_config.txt`.
The data is copied to `~/rgcn/ogbn-mag` on each of the remote machines. `--rel_data_path` specifies the location of the partitioned data (generated in Step 1). `--part_config`
specifies the location of the partitioned data in the local machine (a user only needs to specify
the location of the partition configuration file).
```bash
python3 ~/dgl/tools/copy_partitions.py --ip_config ip_config.txt \
			--workspace ~/rgcn --rel_data_path data \
			--part_config data/ogbn-mag.json
```

**Note**: users need to make sure that the master node has right permission to ssh to all the other nodes.

Users need to copy the training script to the workspace directory on remote machines as well.

### Step 3: Launch distributed jobs

DGL provides a script to launch the training job in the cluster. `part_config` and `ip_config`
specify relative paths to the path of the workspace.

```bash
python3 ~/dgl/tools/launch.py \
--workspace ~/rgcn/ \
--num_trainers 1 \
--num_servers 1 \
--num_samplers 4 \
--part_config data/ogbn-mag.json \
--ip_config ip_config.txt \
"python3 entity_classify_dist.py --graph-name ogbn-mag --dataset ogbn-mag --fanout='25,25' --batch-size 256 --n-hidden 64 --lr 0.01 --eval-batch-size 8 --low-mem --dropout 0.5 --use-self-loop --n-bases 2 --n-epochs 3 --layer-norm --sparse-embedding --ip-config ip_config.txt --num-workers 4 --num-servers 1"
```

## Distributed code runs in the standalone mode

The standalone mode is mainly used for development and testing. The procedure to run the code is much simpler.

### Step 1: graph construction.
When testing the standalone mode of the training script, we should construct a graph with one partition.
```bash
python3 partition_graph.py --dataset ogbn-mag --num_parts 1
```

### Step 2: run the training script
```bash
python3 entity_classify_dist.py --graph-name ogbn-mag  --dataset ogbn-mag --fanout='25,25' --batch-size 256 --n-hidden 64 --lr 0.01 --eval-batch-size 8 --low-mem --dropout 0.5 --use-self-loop --n-bases 2 --n-epochs 3 --layer-norm --ip-config ip_config.txt --conf-path 'data/ogbn-mag.json' --standalone
```