#!/bin/bash
#set -x
if [ $# -ne 1 ]; then
	echo "Usage: $0 input_file"
	exit 1
fi

split_width=60000
input_dir=patch_input
rm -rf $input_dir
mkdir -p $input_dir
output_dir=patch_output
rm -rf $output_dir
mkdir -p $output_dir
tmp_dir=patch_tmp
rm -rf $tmp_dir
mkdir -p $tmp_dir

image_storage_dir=image/
rm -rf $image_storage_dir
mkdir -p $image_storage_dir

cnt=0
total_lines=`wc -l $1 | cut -d " " -f1`
while [[ $cnt -lt $total_lines ]]; do
	(( tmp_int = cnt + split_width ))
	real_width=$split_width
	if [ $tmp_int -gt $total_lines ]; then
		(( real_width = total_lines % split_width ))
	fi
	#echo "head -$tmp_int $1 | tail -$real_width > $input_dir/file_$cnt"
	head -$tmp_int $1 | tail -$real_width > $input_dir/file_$cnt
	nohup python -u generate_base64.py $input_dir/file_$cnt $output_dir/file_$cnt $image_storage_dir 1> $tmp_dir/log_$cnt 2> $tmp_dir/error_$cnt &
	cnt=$tmp_int
done

# 防止立马判断增加睡眠时间
sleep 5

cnt=`ps aux | grep "python -u generate_base64.py" | wc -l`
while [[ $cnt -ne 1 ]]; do
	sleep 5
	echo 'please wait [python -u generate_base64.py]'
	cnt=`ps aux | grep "python -u generate_base64.py" | wc -l`
done

#TODO(your code)
