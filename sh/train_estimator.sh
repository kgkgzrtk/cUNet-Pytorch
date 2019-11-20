#!/bin/bash
NAME=${1:-noname-estimator}
echo CUDA_DEVICE_ORDER="PCI_BUS_ID" > .env
echo CUDA_VISIBLE_DEVICES=${2:-0} >> .env
pipenv run python estimator.py\
    --pkl_path "data_pkl/flickr_offset_under05.pkl"\
    --image_root "/mnt/fs2/2019/Takamuro/db/photos_usa_2016"\
    --save_path cp/estimator/${NAME}\
    --num_epoch 100\
    --num_worker 8\
    --batch_size 32\
    --name $NAME\
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
pipenv run tensorboard --logdir ${check[-1]} --port 8080 --bind-all > /dev/null 2>&1
