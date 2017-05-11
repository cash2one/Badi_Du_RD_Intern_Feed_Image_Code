#!/bin/bash

RANK_ID=$OMPI_COMM_WORLD_RANK
LOCAL_RANK_ID=$OMPI_COMM_WORLD_LOCAL_RANK

#hosts='./test_hosts'
hosts=./hosts-$SLURM_JOB_ID-$RANK_ID
echo -n "" > $hosts

echo "SLURM_JOB_NODELIST:"$SLURM_JOB_NODELIST
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
        host_id=$i
        break
    fi
done

ps_num=2
host_workers=2
task_id=$(($host_id * $host_workers + $LOCAL_RANK_ID))
if [ $RANK_ID -ne $task_id ];then
    echo "rank_id:"$RANK_ID", but need:"$task_id
    exit 1
fi

if [ $host_id -lt $ps_num -a $LOCAL_RANK_ID -eq 0 ];then
    echo "start ps: $local_ip, rank id: $RANK_ID, task id: $host_id" 
    sleep 10
    nohup sh train.sh $hosts ps $host_id $ps_num $host_workers 2>&1 > ./log/ps_${host_id}.out &
fi

echo "start worker: $local_ip, rank id: $RANK_ID, local rank id: $LOCAL_RANK_ID" 
sleep 10
nohup sh train.sh $hosts worker $task_id $ps_num $host_workers 2>&1 > ./log/worker_${task_id}.out
