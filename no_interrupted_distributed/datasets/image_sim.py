# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the pairwise image dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import parsing_ops

from datasets import tf_kv_decoder
from nets import losses_factory 

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

_FILE_PATTERN = '%s-part-*'

SPLITS_TO_SIZES = {'train': 120195, 'validation': 1000}

_NUM_CLASSES = 512

_NUM_SHARDS = 10

_ITEMS_TO_DESCRIPTIONS = {
  'image_cur': 'A color image of cur.', 
  'image_pos': 'A color image of pos.',
  'image_neg': 'A color image of neg.',
}

def data_provider_serving(preprocessing_fn, default_batch_size=None):
  feature_spec = {
    'pair_id': tf.FixedLenFeature((), tf.string, default_value=''),
    'image_cur': tf.FixedLenFeature((), tf.string, default_value=''),
    'image_item': tf.FixedLenFeature((), tf.string, default_value=''),
  }
  
  serialized_tf_example = tf.placeholder(dtype=dtypes.string,
                                                  shape=[default_batch_size],
                                                  name='input_example_tensor')
  tensors = {'examples': serialized_tf_example}
  examples = parsing_ops.parse_example(serialized_tf_example, feature_spec)
  images_cur = examples['image_cur']
  images_item = examples['image_item']
  pairs_id = examples['pair_id']
  def convert_image(images):
    images_arr = tf.split(images, num_or_size_splits=[tf.shape(images)[0]])
    feature_arr = []
    for img_str in images_arr:
      img_str = tf.reshape(img_str, [])
      img_tensor = tf.image.decode_image(img_str, channels=3)
      img_tensor = preprocessing_fn(img_tensor, FLAGS.eval_image_height, FLAGS.eval_image_width)
      img_tensor = tf.expand_dims(img_tensor, 0)
      feature_arr.append(img_tensor)

    return tf.concat(feature_arr, 0)

  images_cur = convert_image(images_cur)
  images_item = convert_image(images_item)
  return [images_cur, images_item], [pairs_id, tensors]

def data_provider_eval(dataset, preprocessing_fn):
  provider = slim.dataset_data_provider.DatasetDataProvider(
    dataset,
    shuffle=False,
    num_readers=FLAGS.num_readers,
    num_epochs=FLAGS.num_epochs_input,
    common_queue_capacity=2 * FLAGS.batch_size,
    common_queue_min=1 * FLAGS.batch_size)
  #for eval data: pair_id, image_cur, image_item
  [pair_id, image_cur, image_item] = provider.get(['image_cur', 'image_pos', 'image_neg'])
      
  image_none = tf.zeros([FLAGS.eval_image_height, FLAGS.eval_image_width, 3], tf.float32)
  def proc():
    good_sample = tf.convert_to_tensor(True, dtype=dtypes.bool)
    img_cur = tf.image.decode_image(image_cur, channels=3)
    img_item = tf.image.decode_image(image_item, channels=3)
    img_cur = preprocessing_fn(img_cur, FLAGS.eval_image_height, FLAGS.eval_image_width)
    img_item = preprocessing_fn(img_item, FLAGS.eval_image_height, FLAGS.eval_image_width)
    return good_sample, pair_id, img_cur, img_item
        
  def none():
    good_sample = tf.convert_to_tensor(False, dtype=dtypes.bool)
    return good_sample, pair_id, image_none, image_none  

  good_sample, pair_id, image_cur, image_item = tf.cond(tf.logical_or(
        tf.equal(image_cur, dataset.decoder.default_value()),
        tf.equal(image_item, dataset.decoder.default_value())), 
        none, proc)

  pairs_id, images_cur, images_item = tf.train.maybe_batch(
        [pair_id, image_cur, image_item], keep_input=good_sample,
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)
 
  return [images_cur, images_item], [pairs_id, None] #labels is None now

def data_provider_queue(dataset, preprocessing_fn):
  provider = slim.dataset_data_provider.DatasetDataProvider(
    dataset,
    shuffle=False,
    num_readers=FLAGS.num_readers,
    num_epochs=FLAGS.num_epochs_input,
    common_queue_capacity=20 * FLAGS.batch_size,
    common_queue_min=10 * FLAGS.batch_size)
  [image_cur, image_pos, image_neg] = provider.get(['image_cur', 'image_pos', 'image_neg'])
      
  image_none = tf.zeros([FLAGS.train_image_height, FLAGS.train_image_width, 3], tf.float32)
  def proc():
    good_sample = tf.convert_to_tensor(True, dtype=dtypes.bool)
    img_cur = tf.image.decode_image(image_cur, channels=3)
    img_pos = tf.image.decode_image(image_pos, channels=3)
    img_neg = tf.image.decode_image(image_neg, channels=3)
    img_cur = preprocessing_fn(img_cur, FLAGS.train_image_height, FLAGS.train_image_width)
    img_pos = preprocessing_fn(img_pos, FLAGS.train_image_height, FLAGS.train_image_width)
    img_neg = preprocessing_fn(img_neg, FLAGS.train_image_height, FLAGS.train_image_width)
    return good_sample, img_cur, img_pos, img_neg
        
  def none():
    good_sample = tf.convert_to_tensor(False, dtype=dtypes.bool)
    return good_sample, image_none, image_none, image_none  

  good_sample, image_cur, image_pos, image_neg = tf.cond(tf.logical_or(tf.logical_or(
        tf.equal(image_cur, dataset.decoder.default_value()),
        tf.equal(image_pos, dataset.decoder.default_value())),
        tf.equal(image_neg, dataset.decoder.default_value())), 
        none, proc)

  images_cur, images_pos, images_neg = tf.train.maybe_batch(
        [image_cur, image_pos, image_neg], keep_input=good_sample,
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)
  #labels = slim.one_hot_encoding(
  #    labels, dataset.num_classes - FLAGS.labels_offset)
  batch_queue = slim.prefetch_queue.prefetch_queue(
        [images_cur, images_pos, images_neg], capacity=2 * FLAGS.num_clones)
 
  return batch_queue

####################
# Define the model #
####################
def clone_fn(batch_queue, network_fn, deploy_config):
  """Allows data parallelism by creating multiple clones of network_fn."""
  images_cur, images_pos, images_neg = batch_queue.dequeue()
  images = tf.concat([images_cur, images_pos, images_neg], 0)

  with slim.arg_scope([slim.model_variable, slim.variable],
      device=deploy_config.variables_device()):
    logits, end_points = network_fn(images)

  #############################
  # Specify the loss function 
  ##############################
  cur_logits, pos_logits, neg_logits = tf.split(logits, num_or_size_splits=3, axis=0)
  pair_losses = losses_factory.triplet_cosine_loss(cur_logits, pos_logits, neg_logits, 2.0) 
  tf.losses.compute_weighted_loss(pair_losses) #add loss to GRAPH_KEYS.LOSS
  #tf.losses.softmax_cross_entropy(
  #    logits=logits, onehot_labels=labels,
  #    label_smoothing=FLAGS.label_smoothing, weights=1.0)
  return end_points

def get_split(split_name, dataset_dir, kv_path, reader=None):
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    kv: kv value dict.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  file_pattern = os.path.join(dataset_dir, _FILE_PATTERN % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TextLineReader

  is_training = False
  if split_name == 'train':
    is_training = True
  decoder = tf_kv_decoder.TfKvDecoder(kv_path, is_training)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=None)
