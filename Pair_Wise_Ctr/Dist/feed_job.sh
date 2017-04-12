#! /bin/bash
echo "==============JOB BEGIN============"

#hosts='./test_hosts'
hosts=./hosts-$SLURM_JOB_ID-ps-0
echo -n "" > $hosts

echo "WORKER SLURM_JOB_NODELIST:"$SLURM_JOB_NODELIST
host_list=`echo $SLURM_JOB_NODELIST | awk -F',' '{for(i=1; i<=NF; i++) print $i}'`
ip_list=""
for h in $host_list;do
    ip=`host -i $h | awk '{print $NF}'`
    ip_list=$ip_list" "$ip
    echo "$ip" >> $hosts
done

sh run_multi_train.sh $hosts ps 0 2>&1 > ps_0.out &

/home/HGCP_Program/software-install/openmpi-1.8.5/bin/mpirun feed_worker.sh
echo "===============JOB END============="
