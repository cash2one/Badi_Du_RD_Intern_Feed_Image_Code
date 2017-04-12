#!/bin/bash
#
# This script performs the following operations:
# 1. Fine-tunes a ResNetV1-50 model on the Feed training set.
# 2. Evaluates the model on the Feed validation set.
#
# Usage:
# cd slim
# ./slim/scripts/tf_pairwise_train.sh

export LD_LIBRARY_PATH=/home/work/cuda-8.0/lib64:/home/work/cudnn/cudnn_v5.1/cuda/lib64:$LD_LIBRARY_PATH

TRAIN_DIR=./tmp/feed/resnet_v2_101

# Where the dataset is saved to.
KV_PATH=./tmp/feed/data/feed_ctr

python=/home/slurm/tools/tf_py/bin/python4.8

host_list=`cat $1`
host_arr=($host_list)
host_num=${#host_arr[@]}
#configure
ps_hosts="${host_arr[0]}:9090"  #only one ps now
worker_hosts="${host_arr[0]}:9091" #equal to worker_num

host_workers=2 #each host worker num
worker_gpus=4 #each worker gpu num
worker_num=0
gpu_max=16 #each worker max gpu id
device_list="0,1,2,3"
device_idx=4
for ((i=0; i<$host_num; i++));
do
    for ((j=0; j<$host_workers; j++));do
        worker_num=$(($worker_num+1))
        if [ $i == 0 -a $j == 0 ];then
            continue
        fi
        port=$((9091+$j))
        worker_hosts=$worker_hosts",""${host_arr[$i]}:$port"
        
        worker_device="$device_idx"
        device_idx=$(($device_idx+1))
        for ((k=1; k<$worker_gpus; k++));do
            worker_device=$worker_device","$device_idx
            device_idx=$(($device_idx+1))
        done
        device_list=$device_list"	"$worker_device
        if [ $device_idx == $gpu_max ];then
            device_idx=0
        fi
    done
    device_idx=0
done

device_arr=($device_list)
job_name=$2 #ps or worker
task_id=$3  #0-worker_num-1, 0 for ps
if [ "$job_name" == "ps" ];then
    export CUDA_VISIBLE_DEVICES='' #for ps
else
    export CUDA_VISIBLE_DEVICES=${device_arr[$task_id]} # for worker id
fi

DATASET_DIR=./tmp/feed/data/$task_id # each worker get 1 of worker_num parts

echo "ps_hosts:"$ps_hosts
echo "worker_hosts:"$worker_hosts
echo "device_list:"$device_list
echo "worker_num:"$worker_num
echo "job_name:"$job_name
echo "task_id:"$task_id
echo "CUDA_VISIBLE_DEVICES:"$CUDA_VISIBLE_DEVICES
echo "DATASET_DIR:"$DATASET_DIR

$python tf_pairwise_train.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=feed \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v2_101 \
  --max_number_of_steps=20000 \
  --batch_size=64 \
  --learning_rate=0.001 \
  --save_interval_secs=120 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --file_info=${KV_PATH} \
  --num_clones=${worker_gpus} \
  --ps_hosts="$ps_hosts" \
  --worker_hosts="$worker_hosts" \
  --task=${task_id} \
  --sync_replicas \
  --replicas_to_aggregate=${worker_num} \
  --job_name=${job_name} \
  --num_readers=4 \
  --num_preprocessing_threads=4 \
  --train_image_height=146 \
  --train_image_width=218 \
  --weight_decay=0.00004

