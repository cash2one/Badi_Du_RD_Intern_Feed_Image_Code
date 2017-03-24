set -e

export PATH=/home/work/Python-4.8.2/bin:/home/work/cuda-8.0/bin$PATH
export LD_LIBRARY_PATH=/home/work/Python-4.8.2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/work/cudnn/cudnn_v5.1/cuda/lib64:/home/work/cuda-8.0/lib64:$LD_LIBRARY_PATH

#python tf_queue.py
#python -m pdb encode_queue.py
#python decode_tf_record.py

python -m pdb alexnet_multi_gpu.py

#python /home/work/Python-4.8.2/lib/python2.7/site-packages/tensorflow/tensorboard/tensorboard.py
#--logdir ./model --port 3838
