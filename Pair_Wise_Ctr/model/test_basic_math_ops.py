#coding=gbk
import tensorflow as tf
from tensorflow.python.ops import math_ops

# TODO cos, maximum, to_float, matmul
# tf.reshape tf.matmul
'''
	https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/math_ops.py
'''

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    x = tf.constant([[1, 2, 3.],[4, 5, 6.],[7, 8, 9.]])
    y = tf.constant([10, 0, 4.])
    x0,x1,x2=tf.split(x, num_or_size_splits=3, axis=0)
    print x0.eval()
	#ans_max = math_ops.maximum(x, y)
	#print ans_max.eval()
	#ans_cos = math_ops.cos(x)
	#print ans_cos.eval()
	# not ok, 矩阵的秩必须>=2
	#ans_matmul = tf.matmul(a = x, b = y, transpose_b = True)
	# also ok!
	#z = x * y
	#print z.eval()
	# 两个向量的 cosine运算
	#ans_tensor_cosine = tf.reduce_sum(math_ops.multiply(x, y)) / tf.sqrt(tf.reduce_sum(tf.square(x)) * tf.reduce_sum(tf.square(y)))
	#print ans_tensor_cosine.eval()
    pass
