#!/bin/bash
NAME=${1:-noname-estimator}
echo CUDA_DEVICE_ORDER="PCI_BUS_ID" > .env
echo CUDA_VISIBLE_DEVICES=${2:-0} >> .env
pipenv run python estimator.py\
    --pkl_path "/mnt/fs2/2018/matsuzaki/results/flickr_data/outdoor1100000_withweather_con_rain_less3.pkl"\
    --image_root "/mnt/fs2/2019/Takamuro/db/photos_usa_2016"\
    --save_path cp/estimator/${NAME}\
    --num_epoch 100\
    --num_worker 4\
    --batch_size 32\
    --name $NAME\
