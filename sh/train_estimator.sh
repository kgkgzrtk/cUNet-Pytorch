#!/bin/bash
NAME=${1:-noname-estimator}
CUDA_DEVICE_ORDER="PCI_BUS_ID"
CUDA_VISIBLE_DEVICES=${2:-0}
pipenv run python estimator.py\
    --pkl_path "/mnt/fs2/2018/matsuzaki/results/flickr_data/add_l2norm.pkl"\
    --image_root "/mnt/fs2/2019/Takamuro/db/photos_usa_2016"\
    --save_path cp/estimator/${NAME}\
    --num_worker 2\
    --batch_size 16;
