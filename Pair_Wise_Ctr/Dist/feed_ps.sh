#!/bin/bash

RANK_ID=$OMPI_COMM_WORLD_RANK
LOCAL_RANK_ID=$OMPI_COMM_WORLD_LOCAL_RANK

#hosts='./test_hosts'
hosts=./hosts-$SLURM_JOB_ID-ps-$RANK_ID
echo -n "" > $hosts

echo "PS SLURM_JOB_NODELIST:"$SLURM_JOB_NODELIST
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
task_id=$host_id
if [ $task_id -ge $ps_num ];then
    echo "ignore ps:$local_ip, rank id: $RANK_ID, task id: $task_id"
    exit 0
fi

echo "start ps: $local_ip, rank id: $RANK_ID, task id: $task_id" 
sleep 10
nohup sh run_multi_train.sh $hosts ps $task_id $ps_num $host_workers 2>&1 > ps_${task_id}.out
