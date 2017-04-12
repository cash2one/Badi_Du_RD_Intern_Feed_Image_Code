#!/bin/bash

RANK_ID=$OMPI_COMM_WORLD_RANK
LOCAL_RANK_ID=$OMPI_COMM_WORLD_LOCAL_RANK

echo "RANK_ID:$RANK_ID"
echo "LOCAL_RANK_ID:$LOCAL_RANK_ID"

#hosts='./test_hosts'
hosts=./hosts-$SLURM_JOB_ID-$LOCAL_RANK_ID
echo -n "" > $hosts

echo "WORKER SLURM_JOB_NODELIST:"$SLURM_JOB_NODELIST
host_list=`echo $SLURM_JOB_NODELIST | awk -F',' '{for(i=1; i<=NF; i++) print $i}'`
ip_list=""
for h in $host_list;do
    ip=`host -i $h | awk '{print $NF}'`
    ip_list=$ip_list" "$ip
    echo "$ip" >> $hosts
done

local_host=`hostname`
local_ip=`host -i $local_host | awk '{print $NF}'`
host_id=0

host_arr=($ip_list)
host_num=${#host_arr[@]}
for ((i=0; i<$host_num; i++));
do
    if [ ${host_arr[$i]} == $local_ip ];then
        host_id=i
        break
    fi
done

task_id=$(($host_id * 4 + $LOCAL_RANK_ID))
if [ $RANK_ID -ne $task_id ];then
    echo "rank_id:"$RANK_ID", but need:"$task_id
    exit 1
fi

#sleep 10
echo "start worker: $local_ip"
sh run_multi_train.sh $hosts worker $task_id 2>&1 > worker_${task_id}.out
