#coding=gbk
import tensorflow as tf
from tensorflow.python.ops import math_ops

import numpy as np

'''
float  y =  label[i] > 0 ? 1 : -1;
float  tmp = std::max(1 - y * pred[i], 0);  loss[i] = tmp * tmp;

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/losses/losses_impl.p
变种: http://blog.csdn.net/luo123n/article/details/48878759
loss(q, pt, nt) = max(0, 1 - cos(v(q), v(pt)) + cos(v(q), v(nt)))
'''

# 必须 float32, 建议输入0~1 sigmoid
def my_hinge_loss(p_cosine, n_cosine):
	#p_cosine = tf.to_float(p_cosine)
	#n_cosine = tf.to_float(n_cosine)
	return math_ops.maximum(0., 1. - p_cosine + n_cosine)

if __name__ == '__main__':
	sess = tf.InteractiveSession()
	x = tf.constant([1, 2, 5.])
	y = tf.constant([10, 0, 4.])
	ans_hinge_loss = my_hinge_loss(x, y)
	print ans_hinge_loss.eval()
