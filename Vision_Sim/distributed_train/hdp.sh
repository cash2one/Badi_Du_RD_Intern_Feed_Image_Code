HADOOP_HOME=/home/feedrd/hgcp/software-install/hadoop-v2/hadoop/bin/hadoop


$HADOOP_HOME fs -Dfs.default.name=hdfs://yq01-idl-gpu-offline58.yq01.baidu.com:24206 \
                -Dhadoop.job.ugi=dl,hadoop123  \
                -ls /user/dl/vision_dataset/
#-put ./train_3 /user/dl/vision_dataset/train_3
