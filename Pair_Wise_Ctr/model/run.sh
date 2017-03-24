set -e

export PATH=/home/work/Python-4.8.2/bin:/home/work/cuda-8.0/bin$PATH
export LD_LIBRARY_PATH=/home/work/Python-4.8.2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/work/cudnn/cudnn_v5.1/cuda/lib64:/home/work/cuda-8.0/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=3

#echo $SLURM_NODELIST | awk -F',' '{print $1}'
#function test() {
#  cfg=$1
#  batch_size=$2
#  prefix=$3
#  python network/$cfg --batch_size=$batch_size > logs/${prefix}-1gpu-${batch_size}.log 2>&1
#}

#python -m pdb ./network/deep_ctr.py
#python resnet_main.py
#python -m pdb network/alexnet.py

python /home/work/Python-4.8.2/lib/python2.7/site-packages/tensorflow/tensorboard/tensorboard.py --logdir ./model --port 8080
#mkdir -p logs
#python test_basic_math_ops.py
# alexnet
#test alexnet.py 1 alexnet
#test alexnet.py 2 alexnet
#test alexnet.py 4 alexnet
#test alexnet.py 8 alexnet
#test alexnet.py 16 alexnet
#test alexnet.py 32 alexnet
#test alexnet.py 64 alexnet
#test alexnet.py 128 alexnet
#test alexnet.py 256 alexnet
#test alexnet.py 512 alexnet

# googlenet
#test googlenet.py 1 googlenet
#test googlenet.py 2 googlenet
#test googlenet.py 4 googlenet
#test googlenet.py 8 googlenet
#test googlenet.py 16 googlenet
#test googlenet.py 32 googlenet
#test googlenet.py 64 googlenet
#test googlenet.py 128 googlenet
#test googlenet.py 256 googlenet
#test googlenet.py 512 googlenet
