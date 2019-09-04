#!/bin/bash
pipenv run python train.py\
    --gpu 0\
    --name "cUNet_good_clsf"\
    --pkl_path "/mnt/data2/matsuzaki/repo/data/sepalated_mini_data.pkl"\
    --classifier_path "cp/classifier/res_aug_5_cls/resnet101_95.pt"\
    --batch_size 16\
    --num_workers 4;
