# Transformer in DGL
In this example we implement the [PolyGen](https://arxiv.org/pdf/2002.10880.pdf) in DGL.

The folder contains training module and inferencing module (nucleus sampling) for Transformer and training module for Universal Transformer

## Dependencies

- PyTorch 0.4.1+
- Pytorch3D

## Usage

- For single-gpu training:

    ```
    python examples/pytorch/polygen/train_vertexnet.py --gpus 0 --dataset /home/ubuntu/data/ShapeNetCore.v2/all_file_list_filtered.txt --batch 2 --ckpt-dir /home/ubuntu/logs/
    

    python examples/pytorch/polygen/train_facenet.py --gpus 0 --dataset /home/ubuntu/data/ShapeNetCore.v2/all_file_list_filtered.txt --batch 2 --ckpt-dir /home/ubuntu/logs/

    ```

- For multi-gpu training:

    ```
    python examples/pytorch/polygen/train_vertexnet.py --gpus 0,1,2,3,4,5,6,7 --dataset /home/ubuntu/data/ShapeNetCore.v2/all_file_list_filtered.txt --batch 16 --ckpt-dir /home/ubuntu/logs/
    
    python examples/pytorch/polygen/train_facenet.py --gpus 0,1,2,3,4,5,6,7 --dataset /home/ubuntu/data/ShapeNetCore.v2/all_file_list_filtered.txt --batch 16 --ckpt-dir /home/ubuntu/logs/
    ```


## TODOs
- Training Speed Tuning
- Nucleus Sampling
- Masked Inference
- Data Augmentation
- Learning Rate Scheduling Check
