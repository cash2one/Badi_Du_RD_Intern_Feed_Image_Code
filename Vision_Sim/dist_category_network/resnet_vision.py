# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet Train/Eval module.
"""
import time
import six
import sys
import os
import numpy as np
import input_set
#import resnet_model
import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.framework.python.ops import variables

from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import regularizers

slim = tf.contrib.slim
bottleneck=resnet_v2.bottleneck
'''
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32, 'batch_size')
tf.app.flags.DEFINE_integer('max_steps', 50, 'Number of batches to run')
tf.app.flags.DEFINE_integer('lrn_rate',0.001,'lrn_rate')
tf.app.flags.DEFINE_integer('logits', 329,'Number of out vector')

tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('train_dir', './model',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('log_root', './model',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 4,
                            'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                             """Whether to log device placement.""")

tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('weight_decay', 0.004, 
                          """regularization""")
'''

#Num_examples_per_epoch = 20000

#TOWER_NAME = 'tower'



def cosine_distance(curr , po_ne):
    cosine = tf.div(tf.reduce_sum(math_ops.multiply(curr,po_ne),1,keep_dims=True),\
                tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(curr),1,keep_dims=True),\
                tf.reduce_sum(tf.square(po_ne),1,keep_dims=True))))
    return cosine

def my_resnet_v2(inputs,
                 blocks,
                 num_classes=None,
                 weight_decay=0.0004,
                 global_pool=True,
                 output_stride=None,
                 include_root_block=True,
                 reuse=None,
                 scope=None):
    with variable_scope.variable_scope(
            scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with arg_scope(
                [layers_lib.conv2d, bottleneck, resnet_utils.stack_blocks_dense],
                outputs_collections=end_points_collection):
            net = inputs
            if include_root_block:
                if output_stride is not None:
                    if output_stride % 4 != 0:
                        raise ValueError('The output_stride needs to be a multiple of 4.')
                    output_stride /= 4
                with arg_scope(
                        [layers_lib.conv2d], activation_fn=None, normalizer_fn=None):
                    #net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                    net = layers_lib.conv2d(net, 64, 7, stride=2, rate=1, padding='VALID', scope='conv1')
                    net = layers.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
            net = layers.batch_norm(net, activation_fn=nn_ops.relu, scope='postnorm')
            if global_pool:
                net = math_ops.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            if num_classes is not None:
                net = layers_lib.conv2d(
                        net,
                        num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='mapping')
            net = tf.squeeze(net, [1, 2], name='reduce')
            
            #num_classes, [1, 1],

            #end_points = utils.convert_collection_to_dict(end_points_collection)
            #if num_classes is not None:
            #    end_points['predictions'] = layers.softmax(net, scope='predictions')

            #curr_sim_logits = tf.squeeze(net, [1, 2], name='Curr_Sim_logits')
            #curr_sim_logits = tf.reshape(curr_sim_logits, [-1, 2, num_classes])
            #curr_logits, sim_logits = tf.split(curr_sim_logits, num_or_size_splits=2, axis=1)

            #curr_logits = tf.squeeze(curr_logits, [1], name='Curr_logits')
            #sim_logits = tf.squeeze(sim_logits, [1], name='Sim_logits')

            #curr_sim_cosine = cosine_distance(curr_logits, sim_logits)

            #final_feature = tf.concat([curr_logits, sim_logits, curr_sim_cosine], axis=1)
            
            #final_feature = tf.concat([curr_logits, sim_logits], axis=1)
            #fc1 = layers.fully_connected(final_feature, 512, 
            #                             weights_initializer=initializers.variance_scaling_initializer(), 
            #                             weights_regularizer=regularizers.l2_regularizer(FLAGS.weight_decay),
            #                             reuse=reuse, scope='Fc1')
            #net = layers.fully_connected(net, num_classes, activation_fn=None,
            #                                weights_initializer=initializers.variance_scaling_initializer(),
            #                                weights_regularizer=regularizers.l2_regularizer(FLAGS.weight_decay),
            #                                reuse=reuse, scope='Logits')

            '''
            fc1_w = tf.get_variable(
                    'Fc1_Weight', [final_feature.get_shape()[1], 50],
                    initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
            fc1_b = tf.get_variable('Fc1_Biases', [50],
                    initializer=tf.constant_initializer())
            fc1 = tf.nn.xw_plus_b(final_feature, fc1_w, fc1_b)
            '''
            return net

def my_resnet(inputs,
              num_classes,
              weight_decay,
              global_pool=True,
              output_stride=None,
              reuse=None,
              scope='resnet_v2'):
    blocks = [
         resnet_utils.Block('block1', bottleneck,
                 [(256, 64, 1)] * 3 + [(256, 64, 2)]),
         resnet_utils.Block('block2', bottleneck,
                 [(512, 128, 1)] * 4 + [(512, 128, 2)]),
         resnet_utils.Block('block3', bottleneck,
                 [(1024, 256, 1)] * 23 + [(1024, 256, 2)]),
         resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return my_resnet_v2(
            inputs,
            blocks,
            num_classes,
            weight_decay,
            global_pool,
            output_stride,
            include_root_block=True,
            reuse=reuse,
            scope=scope)

'''
def my_hinge_loss(p_cosine, n_cosine):
	#p_cosine = tf.to_float(p_cosine)
	#n_cosine = tf.to_float(n_cosine)
	return tf.reduce_mean(math_ops.maximum(0., 1. - p_cosine + n_cosine),name="Loss_mean")


def cosine_distance(curr , po_ne):
    cosine = tf.div(tf.reduce_sum(math_ops.multiply(curr,po_ne),1,keep_dims=True),\
                tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(curr),1,keep_dims=True),\
                tf.reduce_sum(tf.square(po_ne),1,keep_dims=True))))
    return cosine
'''
def inference(samples, labels, weights, weight_decay, feature_num=None, is_training=True, reuse_variables=None):
    #with tf.device('/cpu:0'):
        #image_shape = [2*FLAGS.batch_size, FLAGS.image_size_row, FLAGS.image_size_col, 3]
    '''
    curr_positive_negative_images = tf.get_variable(
           'curr_positive_negative_image',
           image_shape,
           initializer=tf.truncated_normal_initializer(
           stddev=0.1, dtype=tf.float32),
           dtype=tf.float32,
           trainable=False)
        
        curr_sim_images = tf.Variable(
               tf.random_normal(image_shape, stddev=0.35),
               name='curr_positive_negative_image',
               trainable=False)
        labels = tf.Variable(
                tf.random_normal([FLAGS.batch_size, 1], stddev=0.35),
                name='label',
                trainable=False)
    '''
    curr_sim_images = samples
    labels = labels
    with arg_scope(resnet_utils.resnet_arg_scope(is_training=is_training, weight_decay=weight_decay)):
        logits = my_resnet(curr_sim_images, feature_num, weight_decay, reuse=reuse_variables, scope='Vision')
        
        #vision_loss = tf.reduce_mean(losses_impl.sigmoid_cross_entropy(labels, logits), name='Vision_loss')
        cross_entropy = losses_impl.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=weights)
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        #vision_loss =losses_impl.sigmoid_cross_entropy(labels, logits)
        #regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #regular_loss = tf.add_n(regularization_losses, name='regular_loss')
        total_loss = cross_entropy_mean
        #total_loss = cross_entropy_mean + regular_loss

        #curr_sim_logits = tf.squeeze(curr_sim_logits, [1, 2], name='Curr_Sim_logits')
        #curr_sim_logits = tf.reshape(curr_sim_logits, [-1, 2, 50])
        #curr_logits, sim_logits = tf.split(curr_sim_logits, num_or_size_splits=2, axis=1)

        #curr_logits = tf.squeeze(curr_logits, [1], name='Curr_logits')
        #sim_logits = tf.squeeze(sim_logits, [1], name='Sim_logits')
        #negative_logits = tf.squeeze(negative_logits, [1], name='Negative_logits')

    #curr_positive_cosine = cosine_distance(curr_logits, positive_logits)
    #curr_negative_cosine = cosine_distance(curr_logits, negative_logits)
    #pair_wise_loss = my_hinge_loss(curr_positive_cosine,curr_negative_cosine)
    #return pair_wise_loss
    '''
    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
              tf.get_default_graph(),
              tfprof_options=tf.contrib.tfprof.model_analyzer.
               TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
    tf.contrib.tfprof.model_analyzer.print_model_analysis(
              tf.get_default_graph(),
              tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)
    '''
    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        #grad = tf.concat(0, grads)
        grad=tf.concat(grads,0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def main(_):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(
                    0.0, dtype=tf.float32),
                trainable=False,
                dtype=tf.float32)
        num_batches_per_epoch = (Num_examples_per_epoch /
                                 FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
        lr = tf.train.exponential_decay(FLAGS.lrn_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        
        tower_grads = []
        reuse_variables = None
        batch_examples, batch_labels = input_set.get_input(FLAGS.train_data_path, FLAGS.batch_size)
        mini_batch_examples = tf.split(batch_examples, num_or_size_splits=FLAGS.num_gpus, axis=0)
        mini_batch_labels = tf.split(batch_labels, num_or_size_splits=FLAGS.num_gpus, axis=0)

        for i in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                    with arg_scope([variables.variable], device='/cpu:0'):
                        loss = inference(mini_batch_examples[i], mini_batch_labels[i], is_training=True, reuse_variables=reuse_variables)
                        reuse_variables = True
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        print i
                        batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                               scope)
                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)

        summaries.append(tf.summary.scalar('Loss', loss))
        summaries.append(tf.summary.scalar('learning_rate', lr))
        
        grads = average_gradients(tower_grads)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        variable_averages = tf.train.ExponentialMovingAverage(0.9,global_step)
        variables_to_average = (tf.trainable_variables() +
                                  tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)
        
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        train_op = tf.group(apply_gradient_op, variables_averages_op,
                              batchnorm_updates_op)
        

        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge(summaries)
        
        init = tf.global_variables_initializer()
        
                                    
        # Start running operations on the Graph.
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=FLAGS.log_device_placement)

        #config.gpu_options.allocator_type = 'BFC'
        #config.gpu_options.per_process_gpu_memory_fraction = 0.60
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)
        sess.run(init)
        summary_writer=tf.summary.FileWriter(FLAGS.train_dir,sess.graph)

        coord = tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        start_time=time.time()

        for step in xrange(1, FLAGS.max_steps+1):
            _, loss_value = sess.run([train_op,loss])
                #if step % 2 == 0:
            print '%f\t%f\t%f' % (loss_value, sess.run(global_step), time.time()-start_time)
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
            if step%1000==0:
                model_name='vision_model_'+str(step)+'.ckpt'
                checkpoint_path = os.path.join(FLAGS.log_root, model_name)
                saver.save(sess, checkpoint_path, global_step=step)

                #summary_str=sess.run(summary_op)
                #summary_writer.add_summary(summary_str,step)
                #print global_step
        duration = time.time() - start_time

        #checkpoint_path = os.path.join(FLAGS.log_root, 'vision_model.ckpt')
        #saver.save(sess, checkpoint_path, global_step=step)

        print duration
        coord.request_stop()
        coord.join(threads)
        sess.close()
    #elif FLAGS.mode == 'eval':
    #    evaluate()


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
