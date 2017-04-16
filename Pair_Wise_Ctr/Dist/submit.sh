#! /bin/bash

sh ~/hgcp/software-install/HGCP_client/bin/qsub_f \
	--hdfs hdfs://yq01-idl-gpu-offline58.yq01.baidu.com:24206 \
	--hdfs-user dl \
	--hdfs-passwd hadoop123 \
	--hdfs-path /user/dl/ \
	--file-dir ./ \
	--job-name feed_ctr \
	--queue-name yq01-feed-gpubox \
	--num-nodes 5 \
	--num-task-pernode 2 \
	--gpu-pnode 16 \
	./feed_job.sh

