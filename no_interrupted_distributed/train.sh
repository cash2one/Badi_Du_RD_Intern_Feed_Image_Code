#!/bin/bash
#
# This script performs the following operations:
# ./train.sh

HADOOP=/home/HGCP_Program/software-install/hadoop-v2/hadoop/bin/hadoop

export LD_LIBRARY_PATH=/home/work/cuda-8.0/lib64:/home/work/cudnn/cudnn_v5.1/cuda/lib64:$LD_LIBRARY_PATH


host_list=`cat $1`
job_name=$2 #ps or worker
task_id=$3  #0-worker_num-1, 0 for ps
ps_num=$4 #num ps, each host one ps
host_workers=$5 #each host worker num, mod gpu num should be 0
host_gpu_num=$6

host_arr=($host_list)
host_num=${#host_arr[@]}
#configure
ps_hosts="${host_arr[0]}:9090"  #only one ps now
worker_hosts="${host_arr[0]}:9091" #equal to worker_num

worker_gpus=$(($host_gpu_num / $host_workers)) #each worker gpu num
worker_num=0
device_list="0"
device_idx=1
for ((k=1; k<$worker_gpus; k++));do
    device_list=$device_list","$device_idx
    device_idx=$(($device_idx+1))
done

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
        if [ $device_idx == $host_gpu_num ];then
            device_idx=0
        fi
    done
    if [ $i -gt 0 -a $i -lt $ps_num ];then
      ps_hosts=$ps_hosts","${host_arr[$i]}":9090"
    fi
    device_idx=0
done

device_arr=($device_list)
if [ "$job_name" == "ps" ];then
    export CUDA_VISIBLE_DEVICES='' #for ps
else
    export CUDA_VISIBLE_DEVICES=${device_arr[$task_id]} # for worker id
    ${HADOOP} fs -Dfs.default.name=hdfs://nj01-nanling-hdfs.dmop.baidu.com:54310 -Dhadoop.job.ugi=mco_userprofile,userprofile@mco -get /app/mco_userprofile/feed-vertical/feed-image/leijinyi_dl_model/feed_sim_network_dataset/omit_train_* ./data/
#${HADOOP} fs -Dfs.default.name=hdfs://nj01-nanling-hdfs.dmop.baidu.com:54310 -Dhadoop.job.ugi=mco_userprofile,userprofile@mco -get /app/mco_userprofile/feed-vertical/feed-image/label_b64_format/part-0${task_id} ./data/

#$python label_transform.py ./data/part-0${task_id}/ ./convert/
fi



#DATASET_DIR=/home/slurm/data/tmp/feed/data/$task_id # each worker get 1 of worker_num parts
#DATASET_DIR=./convert/part-0${task_id}/part-*


#python=/home/slurm/tools/tf_py/bin/python4.8
dataset_name=vision_sim
model_name=resnet_v2_vision_sim
python=/home/slurm/tools/Python-4.8.2/bin/python
DATASET_DIR=./data/omit_train_*
TRAIN_DIR=./output

echo "ps_hosts:"$ps_hosts
echo "worker_hosts:"$worker_hosts
echo "device_list:"$device_list
echo "worker_num:"$worker_num
echo "job_name:"$job_name
echo "task_id:"$task_id
echo "CUDA_VISIBLE_DEVICES:"$CUDA_VISIBLE_DEVICES
echo "DATASET_DIR:"$DATASET_DIR

$python classify.py \
  --dataset_name=$dataset_name \
  --model_name=$model_name \
  --train_dir=${TRAIN_DIR} \
  --train_data_path=${DATASET_DIR} \
  --max_steps=40000 \
  --batch_size=48 \
  --logits=512 \
  --lrn_rate=0.001 \
  --save_interval_secs=360 \
  --save_summaries_secs=360 \
  --log_every_n_steps=100 \
  --num_clones=${worker_gpus} \
  --ps_hosts="$ps_hosts" \
  --worker_hosts="$worker_hosts" \
  --task=${task_id} \
  --sync_replicas \
  --replicas_to_aggregate=${worker_num} \
  --job_name=${job_name} \
  --weight_decay=0.00004
