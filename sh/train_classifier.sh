#!/bin/bash
NAME=${1:-noname-classifier}
CUDA_VISIBLE_DEVICES=${2:-0}
pipenv run python classifier.py\
    --pkl_path "/mnt/data2/matsuzaki/repo/data/sepalated_data.pkl"\
    --save_path cp/classifier/$NAME\
    --batch_size 32;
