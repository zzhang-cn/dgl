python cluster_gcn.py --gpu 0 --dataset ppi --lr 1e-2 --weight-decay 0.0 --psize 50 --batch-size 1 --n-epochs 300 --n-hidden 1024 --n-layers 1 --log-every 100 --use-pp --self-loop --note self-loop-ppi-non-sym-ly3-pp-cluster-2-2-wd-0 --dropout 0.0 --model-type graphsage --use-val --normalize