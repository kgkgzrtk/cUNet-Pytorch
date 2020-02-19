#!/bin/bash
NAME=${1:-cUNet}
GPU=${2:-0}

pipenv run python train.py\
    --gpu $GPU\
    --name $NAME\
    --save_dir "/mnt/fs2/2018/matsuzaki/results/cp/transfer_class"\
    --pkl_path "/mnt/data2/matsuzaki/repo/data/sepalated_data.pkl"\
    --image_root "/mnt/fs2/2018/matsuzaki/dataset_fromnitta/Image/"\
    --estimator_path "/mnt/data2/matsuzaki/repo/weather_transfer/cp/classifier/res_aug_5_cls/resnet101_95.pt"\
    --lr 1e-4\
    --num_epoch 100\
    --batch_size 16\
    --input_size 224\
    --num_workers 8\
    --one_hot
