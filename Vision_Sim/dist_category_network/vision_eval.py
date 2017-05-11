#!/usr/bin/env python
########################################################################
# 
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
'''
File: vision_eval.py
Author: feedrd(feedrd@baidu.com)
Date: 2017/04/07 14:01:26
'''

import time
import six
import sys
import os
import numpy as np
import input_set
import tensorflow as tf
import resnet_vision

from tensorflow.python.ops import math_ops
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses_impl
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import utils

slim = tf.contrib.slim
bottleneck=resnet_v2.bottleneck

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('eval_batch_size', 32, 'batch_size')
tf.app.flags.DEFINE_integer('eval_max_steps', 10, 'Number of batches to run')
#tf.app.flags.DEFINE_string('eval_dir', './eval',
#                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data_path', '/tmp/imagenet_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './model',
                           """Directory where to read model checkpoints.""")

def evaluate():
    with tf.Graph().as_default():
        images, labels = input_set.get_input(FLAGS.eval_data_path, FLAGS.eval_batch_size)
        curr_sim_images = images
        
        with arg_scope(resnet_utils.resnet_arg_scope(is_training=False)):
            logits = resnet_vision.my_resnet(curr_sim_images, num_classes=FLAGS.logits,
                                 reuse=None, scope='Vision')
            predict = tf.sigmoid(logits, name='Predict')

        variable_averages = tf.train.ExponentialMovingAverage(0.9)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        #summary_op = tf.summary.merge_all()
        #graph_def = tf.get_default_graph().as_graph_def()
        #summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
        #                                 graph_def=graph_def)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                #if os.path.isabs(ckpt.model_checkpoint_path):
                #    saver.restore(sess, ckpt.model_checkpoint_path)
                #else:
                saver.restore(sess, ckpt.model_checkpoint_path)
                
                #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                #print('Successfully loaded model from %s at step=%s.' %
                #        (ckpt.model_checkpoint_path, global_step))
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                            start=True))
                start_time = time.time()
                step = 0
                total_prediction, correct_prediction = 0, 0
                while step < FLAGS.eval_max_steps and not coord.should_stop():
                    predictions, truth = sess.run([predict, labels])
                    step += 1
                    predictions = predictions >0.5
                    correct_prediction += np.sum(truth == predictions)
                    total_prediction += predictions.shape[0]


                precision = 1.0 * correct_prediction / total_prediction

                print '%f\t%f\t%f\t%f' % (correct_prediction, total_prediction, precision, time.time()-start_time)
            except Exception as e:
                coord.request_stop(e)
            
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


if __name__ == '__main__':
    evaluate()
