#!/bin/bash
NAME=${1:-noname-estimator}
echo CUDA_DEVICE_ORDER="PCI_BUS_ID" > .env
echo CUDA_VISIBLE_DEVICES=${2:-0} >> .env
pipenv run python estimator.py\
    --pkl_path "data_pkl/flickr_offset_under05.pkl"\
    --image_root "/mnt/fs2/2019/Takamuro/db/photos_usa_2016"\
    --save_path cp/estimator/${NAME}\
    --num_worker 8\
    --batch_size 16;
