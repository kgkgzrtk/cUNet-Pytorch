#!/bin/bash
pipenv run python demo.py\
    --gpu $1\
    --cp_path "/mnt/fs2/2018/matsuzaki/results/cp/cUNet_wd20/cUNet_wd20_e0002.pt"\
    --input_dir "/mnt/fs2/2018/matsuzaki/data/input"\
    --output_dir "/mnt/fs2/2018/matsuzaki/data/output"\
    --input_size 64
