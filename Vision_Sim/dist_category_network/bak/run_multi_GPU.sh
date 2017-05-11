set -e

export PATH=/home/work/Python-4.8.2/bin:/home/work/cuda-8.0/bin$PATH
export LD_LIBRARY_PATH=/home/work/Python-4.8.2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/work/cudnn/cudnn_v5.1/cuda/lib64:/home/work/cuda-8.0/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

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

python resnet_vision.py --train_data_path=/home/ssd2/category_network/dataset/convert/part-05/part-* \
                        --num_gpus=8 --batch_size=256 --train_dir=./model --max_steps=10000

#python vision_eval.py --eval_data_path=./dataset/omit_train_b64 --eval_batch_size=60 --eval_max_steps=200 --checkpoint_dir=./model 

#python /home/work/Python-4.8.2/lib/python2.7/site-packages/tensorflow/tensorboard/tensorboard.py --logdir ./model --port 8080
#mkdir -p logs
#python -m pdb test_basic_math_ops.py


