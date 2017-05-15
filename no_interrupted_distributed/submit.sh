#! /bin/bash
#chown slurm:slurm hdp.sh
#sh hdp.sh


host_num=2
ps_num=2
host_workers=2
host_gpu_num=8
job_name=feed_vision_sim
queue_name=yq01-hic-p40

sed -i "s#^ps_num=.*#ps_num=$ps_num#g" ./worker.sh
sed -i "s#^host_workers=.*#host_workers=$host_workers#g" ./worker.sh
sed -i "s#^host_gpu_num=.*#host_gpu_num=$host_gpu_num#g" ./worker.sh



sh ~/.hgcp/software-install/HGCP_client/bin/qsub_f \
	--hdfs hdfs://nj01-nanling-hdfs.dmop.baidu.com:54310 \
	--hdfs-user mco_userprofile \
	--hdfs-passwd userprofile@mco \
	--hdfs-path /app/mco_userprofile/feed-vertical/feed-image/leijinyi_dl_model/tmp/ \
	--file-dir ./ \
	--job-name $job_name \
        --submitter leijinyi \
	--queue-name $queue_name  \
	--num-nodes $host_num \
	--num-task-pernode $host_workers \
	--gpu-pnode $host_gpu_num \
	./job.sh

