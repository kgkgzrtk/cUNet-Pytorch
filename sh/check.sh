#!/bin/bash
pipenv run python demo.py\
    --gpu $1\
    --cp_path "/mnt/fs2/2018/matsuzaki/results/cp/cUNet_1010/cUNet_1010_e0005.pt"\
    --input_dir "/mnt/fs2/2018/matsuzaki/data/input"\
    --output_dir "/mnt/fs2/2018/matsuzaki/data/output"\
    --input_size 224\
    --batch_size 16
