set -e

export PATH=/home/work/Python-4.8.2/bin:/home/work/cuda-8.0/bin$PATH
export LD_LIBRARY_PATH=/home/work/Python-4.8.2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/work/cudnn/cudnn_v5.1/cuda/lib64:/home/work/cuda-8.0/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1,2

#export CUDA_VISIBLE_DEVICES=14,15
#python -m pdb test_input.py

#echo $SLURM_NODELIST | awk -F',' '{print $1}'
#function test() {
#  cfg=$1
#  batch_size=$2
#  prefix=$3
#  python network/$cfg --batch_size=$batch_size > logs/${prefix}-1gpu-${batch_size}.log 2>&1
#}

#python resnet_main.py
#python -m pdb network/alexnet.py

dataset_name=vision_sim
model_name=resnet_v2_vision_sim

python -m pdb classify.py \
           --dataset_name=$dataset_name \
           --model_name=$model_name \
           --train_dir=./output \
           --train_data_path=./data/omit_train_b64 \
           --max_steps=2000  \
           --batch_size=24 \
           --lrn_rate=0.001 \
           --save_interval_secs=120 \
           --save_summaries_secs=120 \
           --log_every_n_steps=100 \
           --num_clones=2 \
           --ps_hosts="10.104.18.15:9090"  \
           --worker_hosts="10.104.18.15:9091,10.104.18.15:9092" \
           --task=0 \
           --sync_replicas \
           --replicas_to_aggregate=2 \
           --job_name=worker \
           --weight_decay=0.0004
#python resnet_vision.py --train_data_path=./dataset/omit_train_b64 --num_gpus=4 --batch_size=128 --train_dir=./model --max_steps=10

#python vision_eval.py --eval_data_path=./dataset/omit_train_b64 --eval_batch_size=60 --eval_max_steps=200 --checkpoint_dir=./model 

#python /home/work/Python-4.8.2/lib/python2.7/site-packages/tensorflow/tensorboard/tensorboard.py --logdir ./model --port 8080
#mkdir -p logs
#python -m pdb test_basic_math_ops.py


