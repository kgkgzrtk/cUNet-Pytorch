#!/bin/bash
NAME=${1:-cUNet}
GPU=${2:-0}

pipenv run python train.py\
    --gpu $GPU\
    --name $NAME\
    --save_dir "/mnt/fs2/2018/matsuzaki/results/cp"\
    --pkl_path "/mnt/fs2/2018/matsuzaki/results/flickr_data/df_con_train.pkl"\
    --image_root "/mnt/fs2/2019/Takamuro/db/photos_usa_2016_outdoor"\
    --estimator_path "/mnt/data2/matsuzaki/repo/weather_transfer/cp/estimator/out110_res101_less25_c2/resnet50_10.pt"\
    --lr 1e-4\
    --num_epoch 100\
    --batch_size 16\
    --input_size 224\
    --num_workers 8
: ' -- You can unlock when you want to execute with tensorboard.
PID=$!
cd runs
declare -a check=()
while [ ${#check[@]} -lt 1 ]; do
    sleep 3
    check=(`ls *name-${NAME} -1d| sort -n`)
done
echo "Start tensorboard logdir:${check[-1]}"
pipenv run tensorboard --logdir ${check[-1]} --port 8080 --bind_all > /dev/null 2>&1
trap "kill ${PID}" EXIT
'
