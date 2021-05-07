#!/bin/bash

. /opt/conda/etc/profile.d/conda.sh
conda activate pytorch-ci
GCN_EXAMPLE_DIR="./examples/pytorch/"

function fail {
    echo FAIL: $@
    exit -1
}

function usage {
    echo "Usage: $0 [cpu|gpu]"
}

# check arguments
if [ $# -ne 1 ]; then
    usage
    fail "Error: must specify device"
fi

if [ "$1" == "cpu" ]; then
    dev=-1
elif [ "$1" == "gpu" ]; then
    export CUDA_VISIBLE_DEVICES=0
    dev=0
else
    usage
    fail "Unknown device $1"
fi

export DGLBACKEND=pytorch
export DGL_LIBRARY_PATH=${PWD}/build
export PYTHONPATH=${PWD}/python:$PYTHONPATH
export DGL_DOWNLOAD_DIR=${PWD}

# test

pushd $GCN_EXAMPLE_DIR> /dev/null

python3 pagerank.py || fail "run pagerank.py on $1"
python3 gcn/gcn.py --dataset cora --gpu $dev || fail "run gcn/gcn.py on $1"
python3 lda/lda_model.py || fail "run lda/lda_model.py on $1"

popd > /dev/null
