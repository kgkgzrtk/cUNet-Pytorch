#!/bin/bash
CUDA_VISIBLE_DEVICES=${1:-0}
pipenv run python classifier.py\
    --pkl_path "/mnt/data2/matsuzaki/repo/data/sepalated_data.pkl"\
    --save_path "cp/classifier/res_aug_5_cls_large"\
    --batch_size 32;
