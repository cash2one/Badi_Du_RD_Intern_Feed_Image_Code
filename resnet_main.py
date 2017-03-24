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
slim = tf.contrib.slim
bottleneck=resnet_v2.bottleneck


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch_size')
tf.app.flags.DEFINE_integer('lrn_rate',0.001,'lrn_rate')
tf.app.flags.DEFINE_integer('logits', 50,'Number of out vector')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 200, 'Image side length.')
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
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                             """Whether to log device placement.""")
def my_resnet(inputs,
              num_classes=None,
              global_pool=True,
              output_stride=None,
              reuse=None,
              scope='resnet_v2'):
    blocks = [
         resnet_utils.Block('block1', bottleneck,
                 [(256, 64, 1)] * 1 + [(256, 64, 2)])
         #resnet_utils.Block('block2', bottleneck,
         #        [(512, 128, 1)] * 1 + [(512, 128, 2)])
         #resnet_utils.Block('block3', bottleneck,
         #        [(1024, 256, 1)] * 1 + [(1024, 256, 2)])
         #resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 1)
    ]
    return resnet_v2.resnet_v2(
            inputs,
            blocks,
            num_classes,
            global_pool,
            output_stride,
            include_root_block=True,
            reuse=reuse,
            scope=scope)


def my_hinge_loss(p_cosine, n_cosine):
	#p_cosine = tf.to_float(p_cosine)
	#n_cosine = tf.to_float(n_cosine)
	return tf.reduce_mean(math_ops.maximum(0., 1. - p_cosine + n_cosine),name="Loss_mean")


def cosine_distance(curr , po_ne):
    cosine = tf.div(tf.reduce_sum(math_ops.multiply(curr,po_ne),1,keep_dims=True),\
                tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(curr),1,keep_dims=True),\
                tf.reduce_sum(tf.square(po_ne),1,keep_dims=True))))
    return cosine

def inference():
    with tf.device('/cpu:0'):
        image_shape = [3*FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3]
        curr_positive_negative_images = tf.get_variable(
           'curr_positive_negative_image',
           image_shape,
           initializer=tf.truncated_normal_initializer(
           stddev=0.1, dtype=tf.float32),
           dtype=tf.float32,
           trainable=False)

    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        #curr_positive_negative_logits, end_points = resnet_v2.resnet_v2_50(curr_positive_negative_images, num_classes=FLAGS.logits, scope='pair_wise_rank')
        curr_positive_negative_logits, end_points = my_resnet(curr_positive_negative_images, num_classes=FLAGS.logits, scope='pair_wise_rank')
        curr_positive_negative_logits = tf.squeeze(curr_positive_negative_logits, [1, 2], name='Curr_Positive_Negative_logits')
        curr_positive_negative_logits = tf.reshape(curr_positive_negative_logits, [-1, 3, 50])
        curr_logits, positive_logits, negative_logits = tf.split(curr_positive_negative_logits, num_or_size_splits=3, axis=1)

        curr_logits = tf.squeeze(curr_logits, [1], name='Curr_logits')
        positive_logits = tf.squeeze(positive_logits, [1], name='Positive_logits')
        negative_logits = tf.squeeze(negative_logits, [1], name='Negative_logits')

    curr_positive_cosine = cosine_distance(curr_logits, positive_logits)
    curr_negative_cosine = cosine_distance(curr_logits, negative_logits)
    pair_wise_loss = my_hinge_loss(curr_positive_cosine,curr_negative_cosine)
    return pair_wise_loss
    #param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
    #          tf.get_default_graph(),
    #          tfprof_options=tf.contrib.tfprof.model_analyzer.
    #           TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    #sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
    #tf.contrib.tfprof.model_analyzer.print_model_analysis(
    #          tf.get_default_graph(),
    #          tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

    '''    
  curr_model = resnet_model.ResNet(hps, curr_images, FLAGS.mode)
  curr_model.build_graph()

  positive_model = resnet_model.ResNet(hps, positive_images, FLAGS.mode)
  positive_model.build_graph()

  negitive_model = resnet_model.ResNet(hps, negitive_images, FLAGS.mode)
  negitive_model.build_graph()


  curr_positive_cosine = cosine_distance(curr_model.logits, positive_model.logits)
  curr_negitive_cosine = cosine_distance(curr_model.logits, negitive_model.logits)
  pair_wise_loss = my_hinge_loss(curr_positive_cosine,curr_negitive_cosine)


  param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.
          TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  
  sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
  
  tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)
  
  truth = tf.argmax(model.labels, axis=1)
  predictions = tf.argmax(model.predictions, axis=1)
  precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

  summary_hook = tf.train.SummarySaverHook(
      save_steps=100,
      output_dir=FLAGS.train_dir,
      summary_op=tf.summary.merge([model.summaries,
                                   tf.summary.scalar('Precision', precision)]))

  logging_hook = tf.train.LoggingTensorHook(
      tensors={'step': model.global_step,
               'loss': model.cost,
               'precision': precision},
      every_n_iter=100)

  class _LearningRateSetterHook(tf.train.SessionRunHook):
    """Sets learning_rate based on global step."""

    def begin(self):
      self._lrn_rate = 0.1

    def before_run(self, run_context):
      return tf.train.SessionRunArgs(
          model.global_step,  # Asks for global step value.
          feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

    def after_run(self, run_context, run_values):
      train_step = run_values.results
      if train_step < 40000:
        self._lrn_rate = 0.1
      elif train_step < 60000:
        self._lrn_rate = 0.01
      elif train_step < 80000:
        self._lrn_rate = 0.001
      else:
        self._lrn_rate = 0.0001

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.log_root,
      hooks=[logging_hook, _LearningRateSetterHook()],
      chief_only_hooks=[summary_hook],
      # Since we provide a SummarySaverHook, we need to disable default
      # SummarySaverHook. To do that we set save_summaries_steps to 0.
      save_summaries_steps=0,
      config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
    while not mon_sess.should_stop():
      mon_sess.run(model.train_op)
    '''

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


def main(_):
  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:3'
  else:
    raise ValueError('Only support 0 or 1 gpu.')

  with tf.device(dev):
    if FLAGS.mode == 'train':
        loss = inference()
        opt = tf.train.MomentumOptimizer(FLAGS.lrn_rate, 0.9)
        grads = opt.compute_gradients(loss)
        global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(
                    0.0, dtype=tf.float32),
                trainable=False,
                dtype=tf.float32)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        train_op = tf.group(apply_gradient_op)
            # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(0.9,global_step)
        variables_averages_op = variable_averages.apply(
             tf.trainable_variables())

            # Build an initialization operation.
        init = tf.global_variables_initializer()

            # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=FLAGS.log_device_placement))
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables())
        summary_writer=tf.summary.FileWriter(FLAGS.train_dir,sess.graph)

        coord = tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        start_time=time.time()

        for step in xrange(FLAGS.batch_size):
            _, loss_value = sess.run([train_op,loss])
            #if step % 2 == 0:
            print '%f\t%f\t%f' % (loss_value,sess.run(global_step), time.time()-start_time)
            if step%100==0:
                model_name='pair_wise_model_'+str(step)+'.ckpt'
                checkpoint_path = os.path.join(FLAGS.log_root, model_name)
                saver.save(sess, checkpoint_path, global_step=step)

                #summary_str=sess.run(summary_op)
                #summary_writer.add_summary(summary_str,step)
                #print global_step
        duration = time.time() - start_time

        #checkpoint_path = os.path.join(FLAGS.log_root, 'pair_wise_model.ckpt')
        #saver.save(sess, checkpoint_path, global_step=step)

        print duration
        coord.request_stop()
        coord.join(threads)
        sess.close()
    elif FLAGS.mode == 'eval':
        evaluate()


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
