#!/bin/bash
NAME=${1:-noname-classifier}
echo CUDA_DEVICE_ORDER="PCI_BUS_ID" > .env
echo CUDA_VISIBLE_DEVICES=${2:-0} >> .env
pipenv run python classifier.py\
    --pkl_path "/mnt/data2/matsuzaki/repo/data/sepalated_data.pkl"\
    --save_path cp/classifier/$NAME\
    --name $NAME\
    --mode V\
    --batch_size 32;
