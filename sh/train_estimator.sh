#!/bin/bash
NAME=${1:-noname-estimator}
CUDA_DEVICE_ORDER="PCI_BUS_ID"
CUDA_VISIBLE_DEVICES=${2:-0}
pipenv run python estimator.py\
    --pkl_path "add_l2norm.pkl"\
    --save_path cp/classifier/${NAME}\
    --batch_size 16;
