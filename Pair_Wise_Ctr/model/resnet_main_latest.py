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
import resnet_model
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import utils
import get_input_new

slim = tf.contrib.slim
bottleneck=resnet_v2.bottleneck

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch_size')
tf.app.flags.DEFINE_integer('max_steps', 20000, 'Number of batches to run')
tf.app.flags.DEFINE_integer('lrn_rate',0.001,'lrn_rate')
tf.app.flags.DEFINE_integer('logits', 50,'Number of out vector')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size_row', 200, 'Image row side length.')
tf.app.flags.DEFINE_integer('image_size_col', 200, 'Image col side length.')
tf.app.flags.DEFINE_string('train_dir', './model',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', './model',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 4,
                            'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                             """Whether to log device placement.""")

tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                           """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                             """Learning rate decay factor.""")


TOWER_NAME = 'tower'

def my_resnet_v2(inputs,
                 blocks,
                 num_classes=None,
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
            '''
            if num_classes is not None:
                net = layers_lib.conv2d(
                        net,
                        num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='logits')
            end_points = utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
                end_points['predictions'] = layers.softmax(net, scope='predictions')
            '''
            fc1 = layers.fully_connected(net, 512, reuse=True, scope='Fc1')
            fc2 = layers.fully_connected(fc1, 512, activation_fn=None, reuse=True, scope='Fc2')
            return fc2

def my_resnet(inputs,
              num_classes=None,
              global_pool=True,
              output_stride=None,
              reuse=None,
              scope='resnet_v2'):
    blocks = [
         resnet_utils.Block('block1', bottleneck,
                 [(256, 64, 1)] * 2 + [(256, 64, 2)]),
         resnet_utils.Block('block2', bottleneck,
                 [(512, 128, 1)] * 3 + [(512, 128, 2)]),
         resnet_utils.Block('block3', bottleneck,
                 [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
         resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return my_resnet_v2(
            inputs,
            blocks,
            num_classes,
            global_pool,
            output_stride,
            include_root_block=True,
            reuse=reuse,
            scope=scope)

#return a int tensor,means avg of a batch loss
def my_hinge_loss(p_cosine, n_cosine):
	#p_cosine = tf.to_float(p_cosine)
	#n_cosine = tf.to_float(n_cosine)
    #[batch_size, 1] -> int
	return tf.reduce_mean(math_ops.maximum(0., 2. - (p_cosine - n_cosine)),name="Loss_mean")


#return shape [batch_size, 1]
def cosine_distance(curr , po_ne):
    cosine = tf.div(tf.reduce_sum(math_ops.multiply(curr,po_ne),1,keep_dims=True),\
                tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(curr),1,keep_dims=True),\
                tf.reduce_sum(tf.square(po_ne),1,keep_dims=True))))
    return cosine

def inference(examples, is_training=True, reuse_variables=None):
    with tf.device('/cpu:0'):
        #image_shape = [3*FLAGS.batch_size, FLAGS.image_size_row, FLAGS.image_size_col, 3]
        #curr_positive_negative_images = tf.Variable(
        #       tf.random_normal(image_shape, stddev=0.35),
        #       name='curr_positive_negative_image',
        #       trainable=False)
        curr_positive_negative_images = examples
    with arg_scope(resnet_utils.resnet_arg_scope(is_training=is_training)):
        curr_positive_negative_logits = my_resnet(curr_positive_negative_images, num_classes=FLAGS.logits, 
                                                              reuse=reuse_variables, scope='pair_wise_rank')
        curr_positive_negative_logits = tf.squeeze(curr_positive_negative_logits, [1, 2], name='Curr_Positive_Negative_logits')
        curr_positive_negative_logits = tf.reshape(curr_positive_negative_logits, [-1, 3, 512])
        curr_logits, positive_logits, negative_logits = tf.split(curr_positive_negative_logits, num_or_size_splits=3, axis=1)

        curr_logits = tf.squeeze(curr_logits, [1], name='Curr_logits')
        positive_logits = tf.squeeze(positive_logits, [1], name='Positive_logits')
        negative_logits = tf.squeeze(negative_logits, [1], name='Negative_logits')

    curr_positive_cosine = cosine_distance(curr_logits, positive_logits)
    curr_negative_cosine = cosine_distance(curr_logits, negative_logits)
    pair_wise_loss = my_hinge_loss(curr_positive_cosine,curr_negative_cosine)
    tf.summary.scalar('Curr_Positive_Cosine', curr_positive_cosine)
    tf.summary.scalar('Curr_Negative_Cosine', curr_negative_cosine)
    tf.summary.scalar('Loss', pair_wise_loss)

    #return pair_wise_loss
    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
              tf.get_default_graph(),
              tfprof_options=tf.contrib.tfprof.model_analyzer.
               TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
    tf.contrib.tfprof.model_analyzer.print_model_analysis(
              tf.get_default_graph(),
              tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)
    return pair_wise_loss

def evaluate(hps):
  """Eval loop."""
  '''
  images, labels = cifar_input.build_input(
      FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode)
  '''

  model = resnet_model.ResNet(hps, images,FLAGS.mode)
  model.build_graph()
  saver = tf.train.Saver()
  #summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  tf.train.start_queue_runners(sess)

  best_precision = 0.0
  while True:
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
      continue
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    total_prediction, correct_prediction = 0, 0
    for _ in six.moves.range(FLAGS.eval_batch_count):
      (summaries, loss, predictions, truth, train_step) = sess.run(
          [model.summaries, model.cost, model.predictions,
           model.labels, model.global_step])

      truth = np.argmax(truth, axis=1)
      predictions = np.argmax(predictions, axis=1)
      correct_prediction += np.sum(truth == predictions)
      total_prediction += predictions.shape[0]

    precision = 1.0 * correct_prediction / total_prediction
    best_precision = max(precision, best_precision)

    precision_summ = tf.Summary()
    precision_summ.value.add(
        tag='Precision', simple_value=precision)
    summary_writer.add_summary(precision_summ, train_step)
    best_precision_summ = tf.Summary()
    best_precision_summ.value.add(
        tag='Best Precision', simple_value=best_precision)
    summary_writer.add_summary(best_precision_summ, train_step)
    summary_writer.add_summary(summaries, train_step)
    tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                    (loss, precision, best_precision))
    summary_writer.flush()

    if FLAGS.eval_once:
      break

    time.sleep(60)


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
        #num_batches_per_epoch = (dataset.num_examples_per_epoch() /
        #                            FLAGS.batch_size)

        #decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
        decay_steps = int(5000)
        lr = tf.train.exponential_decay(FLAGS.lrn_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True)

        opt = tf.train.MomentumOptimizer(lr, 0.9)
        tower_grads = []
        reuse_variables = None
        batch_examples = get_input_new.get_input(FLAGS.train_data_path, FLAGS.batch_size)
        mini_batch_examples = tf.split(batch_examples, num_or_size_splits=FLAGS.num_gpus, axis=0)
        for i in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                    with arg_scope([variables.variable], device='/cpu:0'):
                        loss = inference(mini_batch_examples[i], is_training=True, reuse_variables=reuse_variables)
                        reuse_variables = True
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                               scope)
                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)


        grads = average_gradients(tower_grads)
        
        #summaries = tf.summary.scalar('learning_rate', lr)
        summaries.append(tf.summary.scalar('learning_rate', lr))

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

        #config.gpu_options.per_process_gpu_memory_fraction = 0.60
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)
        sess.run(init)
        summary_writer=tf.summary.FileWriter(FLAGS.train_dir,sess.graph)

        coord = tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        start_time=time.time()

        for step in xrange(FLAGS.max_steps):
            _, loss_value = sess.run([train_op,loss])
                #if step % 2 == 0:
            print '%f\t%f\t%f' % (loss_value, sess.run(global_step), time.time()-start_time)
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
        duration = time.time() - start_time

        print duration
        coord.request_stop()
        coord.join(threads)
        sess.close()
    #elif FLAGS.mode == 'eval':
    #    evaluate()


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
