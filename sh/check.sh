#!/bin/bash
NAME="cunet_flickr_short_soft_new"
pipenv run python demo.py\
    --gpu $1\
    --cp_path "/mnt/fs2/2018/matsuzaki/results/cp/${NAME}/${NAME}_e0030.pt"\
    --input_dir "/mnt/fs2/2018/matsuzaki/data/input_display"\
    --output_dir "/mnt/fs2/2018/matsuzaki/data/output_display"\
    --estimator_path "resnet101_95.pt"\
    --input_size 224\
    --batch_size 4\
    --alpha 2
