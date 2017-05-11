#! /bin/bash
#chown slurm:slurm hdp.sh
#sh hdp.sh
sh ~/hgcp/software-install/HGCP_client/bin/qsub_f \
	--hdfs hdfs://nj01-nanling-hdfs.dmop.baidu.com:54310 \
	--hdfs-user mco_userprofile \
	--hdfs-passwd userprofile@mco \
	--hdfs-path /app/mco_userprofile/feed-vertical/feed-image/leijinyi_dl_model/feed_gpubox_tmp/ \
	--file-dir ./ \
	--job-name feed_vision_category \
	--queue-name yq01-feed-gpubox  \
	--num-nodes 2 \
	--num-task-pernode 2 \
	--gpu-pnode 16 \
	./job.sh

