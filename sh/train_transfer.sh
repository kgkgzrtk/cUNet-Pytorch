#!/bin/bash
NAME=${1:-cUNet}
GPU=${2:-0}

pipenv run python train.py\
    --gpu $GPU\
    --name $NAME\
    --pkl_path "/mnt/data2/matsuzaki/repo/data/sepalated_mini_data.pkl"\
    --classifier_path "cp/classifier/res_aug_5_cls/resnet101_95.pt"\
    --lr 5e-5\
    --num_epoch 20\
    --batch_size 16\
    --input_size 224\
    --num_workers 8\
    &
PID=$!
trap "kill ${PID}" EXIT
cd runs
declare -a check=()
while [ ${#check[@]} -lt 1 ]; do
    sleep 3
    check=(`ls *name-${NAME} -1d| sort -n`)
done
echo "Start tensorboard logdir:${check[-1]}"
pipenv run tensorboard --logdir ${check[-1]} --port 8080 > /dev/null 2>&1
