#!/bin/bash

test_array=(1 2 4 8 16 32 64 128)
train_array=(64 128 256 512)
net_array=('alexnet' 'googlenet')

function get_speed()
{
    net=$1
    batch=$2
    istrain=$3

    [ "$istrain" = "true" ] && key="Forward-" || key="Forward "
    speed=`cat logs/$net-1gpu-$batch.log | grep "$key" | awk '{print $7}'`
    speed=$(echo "$speed * $batch" | bc)
    echo "    $speed samples / sec"
}

for net in ${net_array[@]}
do
    echo "$net test result:"
    for batch in ${test_array[@]}
    do
        get_speed $net $batch false
    done
    echo "$net train result:"
    for batch in ${train_array[@]}
    do
        get_speed $net $batch true
    done
done

