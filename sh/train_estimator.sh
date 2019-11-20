#!/bin/bash
NAME=${1:-noname-estimator}
CUDA_DEVICE_ORDER="PCI_BUS_ID"
CUDA_VISIBLE_DEVICES=${2:-0}
pipenv run python estimator.py\
    --pkl_path "data_pkl/flickr_offset_under05.pkl"\
    --image_root "/mnt/fs2/2019/Takamuro/db/photos_usa_2016"\
    --save_path cp/estimator/${NAME}\
    --num_worker 4\
    --batch_size 16;
