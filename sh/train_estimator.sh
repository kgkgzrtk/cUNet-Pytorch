#!/bin/bash
NAME=${1:-noname-classifier}
CUDA_VISIBLE_DEVICES=${2:-0}
pipenv run python estimator.py\
    --pkl_path "/mnt/fs2/2018/matsuzaki/results/flickr_data/add_l2norm.pkl"\
    --save_path cp/classifier/${NAME}\
    --batch_size 32;
