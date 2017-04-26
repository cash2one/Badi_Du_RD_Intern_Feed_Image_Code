#!/bin/bash


source ./conf.sh
source ./public_method.sh

today=`date '+%Y%m%d'`
one_day_ago=`date -d'2 days ago' '+%Y%m%d'`
two_day_ago=`date -d'2 days ago' '+%Y%m%d'`
#one_day_ago=20160705
#two_day_ago=20160704

#ROOT_PATH=/app/st/wise-tc/sb_feed/merge_usermodel
#HOMEPAGE_CUID_USER_MODEL_BASE_PATH=$ROOT_PATH/cuid_homepage_model
#HOMEPAGE_UID_USER_MODEL_BASE_PATH=$ROOT_PATH/uid_homepage_model
          
#log_file=`pwd`/../log/$today".log.scp"
log_file=`pwd`/$today".log.scp"

s1_UGI=ps-rank,rank-hadoop
s2_UGI=mco_userprofile,userprofile@mco
d_UGI=wise-tc,TBYzNICrQFPMKX1u


echo "hadoop home:"$HADOOP_HOME_SB >> $log_file
echo "java home:"$JAVA_HOME >> $log_file


#user_id_cp_path=`$HADOOP_HOME/bin/hadoop fs -ls /user/mco_userprofile/feed_video_online/shortmv_user_model_mining/common/merge_all | grep $one_day_ago | awk -F" " '{print $8}'|head -1` >> $log_file 2>&1
#user_id_cp_path="/user/mco_userprofile/feed_video_online/tools"
#copy_path="shortmv_content_model/merge_full/20160919"
#copy_path="shortmv_content_model/merge_full/20161014"
#copy_path="user_ctr/user_ctr_model/sv_user_ctr_20160829_20161017"
#copy_path="user_ctr/user_ctr_model/sv_user_ctr_20160831_20161019"
#copy_path="shortmv_content_model/merge_full/20161122"
#copy_path="user_ctr/user_item_join/20161121"
copy_path="shortmv_user_model_mining/usermodel/usermodel_sv_20161014_20161202"
#shortmv_content_model/merge_full/20161013
#url2id_dict
#user_id_cp_path="${SRC_PREFIX}/shortmv_content_model/url2id_dict/20160919"
COPY_SRC="${AFS_PATH_PREFIX}/${copy_path}"
#dest_path="/app/idl/wuhonghuan/feed_video_online"
#dest_path="/app/st/wise-tc/wuhonghuan/feed_video_online"
COPY_DEST="${WISE_PATH_PREFIX}/${copy_path}"
echo $COPY_SRC
echo $COPY_DEST
echo $AFS_PATH_PREFIX



echo "user_id_cp_path:"$user_id_cp_path >> $log_file

is_failed="0"
echo "------------------------start copy user id model" >> $log_file
if [[ $COPY_SRC != "" ]];then
	HADOOP_RUN="$HADOOP_HOME_SB/bin/hadoop"
	HADOOP_HUN="$HADOOP_HOME_HUN/bin/hadoop"
#   $HADOOP_HOME_SB/bin/hadoop distcp \
    $HADOOP_HUN distcp \
        -D mapred.job.priority="VERY_HIGH" \
        -D mapred.job.map.capacity=200 \
        -D distcp.map.speed.kb=40000 \
        -su ${s2_UGI} -du ${d_UGI} $COPY_SRC $COPY_DEST >> $log_file 2>&1
	echo $COPY_SRC
	echo $COPY_DEST
    if [ $? -ne 0 ];then
        is_failed="1"
    fi
fi

content=`cat $log_file`
echo "$content"
#send_succ_mail "distcp_user_model [date:${today}] [is_failed=${is_failed}]" "$content"
#send_succ_telephone "distcp_user_model [date:${today}] [is_failed=${is_failed}]" "$log_file"

