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
"""Provides data for the feed dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import base64
import struct
from zlib import crc32
import ctypes

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import parsing_ops
from tensorflow.contrib.slim.python.slim.data import parallel_reader
from tensorflow.contrib.slim.python.slim.data import data_decoder

from nets import losses_factory 
from kvrecord import tf_record_coder
from datasets import generator_io

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

_FILE_PATTERN = '%s-part-*'

SPLITS_TO_SIZES = {'train': 120195, 'validation': 1000}

_NUM_CLASSES = 2

_NUM_SHARDS = 10

_NUM_DIMENSION = 64000

_ITEMS_TO_DESCRIPTIONS = {
  'label': 'label of instance', 
  'feature': 'index:value of instance feature column',
}

_float_feature = lambda v: tf.train.Feature(float_list=tf.train.FloatList(value=v))
_int_feature = lambda v: tf.train.Feature(int64_list=tf.train.Int64List(value=v))
_bytes_feature = lambda v: tf.train.Feature(bytes_list=tf.train.BytesList(value=v))

def parse_oneline(line):
  l = line.rstrip().split()
  label = int(l[0])
    
  indexes = []
  values = []
  #notice most libsvm will use 1 as start index
  index = 1
  for item in l[1:]:
    if ':' in item:
      k, v = item.split(':')
      index = int(k)
      value = float(v)
    else:
      value = float(item)
    indexes.append(index % _NUM_DIMENSION)
    values.append(value)
    index += 1

  example = tf.train.Example(features=tf.train.Features(feature={
      'label': _int_feature([label]),
      'indices': _int_feature(indexes),
      'values': _float_feature(values)
  }))

  return example


def encode_base64(in_file, out_file):
  for line in in_file:
    example = parse_oneline(line)
    value = example.SerializeToString()
    buf = tf_record_coder.encode(value) 
    #out_file.write(buf)
    print(base64.urlsafe_b64encode(buf), file=out_file)

def encode_tfrecord(dataset_dir, split_name, num_per_shard):
  data_sources = os.path.join(dataset_dir, "*")
  data_files = parallel_reader.get_data_files(data_sources)
  tfrecord_dir = "%s_tfrecord" % dataset_dir
  tf.gfile.MakeDirs(tfrecord_dir)
  with tf.Graph().as_default():
    with tf.Session('') as sess:
      shard_id = 0
      output_filename = '%s/%s-part-%05d' % (tfrecord_dir, split_name, shard_id)
      tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)
      cur_num = 0
      shard_id += 1
      for dt in data_files:
        # Read the filename:
        if cur_num >= num_per_shard:
          tfrecord_writer.close()
          output_filename = '%s/%s-part-%05d' % (tfrecord_dir, split_name, shard_id)
          tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)
          cur_num = 0
          shard_id += 1

        with tf.gfile.Open(dt, 'r') as f:
          for line in f:
            example = parse_oneline(line) 
            tfrecord_writer.write(example.SerializeToString())
            cur_num += 1

      tfrecord_writer.close()

def data_provider_stream(in_stream):
  def generator(meta=False):
    if meta:
      yield {
        'keys': ['id', 'label', 'values'],
        'types': [dtypes.string, dtypes.int64, dtypes.float32],
        'shapes': [(), (), [_NUM_DIMENSION]] 
      }
    #format: qe_id \t label 1:1 3:1 5:1 ... 
    line = in_stream.readline()
    while line:
      l = line.rstrip().split('\t')
      id = l[0]
      l = l[1].split()
      label = int(l[0])
      
      values = [0] * _NUM_DIMENSION
      index = 1
      for item in l[1:]:
        if ':' in item:
          k, v = item.split(':')
          index = int(k)
          values[index % _NUM_DIMENSION] = float(v)
        else:
          values[index % _NUM_DIMENSION] = float(item)
        index += 1    
      yield {
        'id': id,
        'label': label,
        'values': values
      }
      line = in_stream.readline()

  input_fn = generator_io.generator_input_fn(
    generator,
    target_key='label',
    batch_size=FLAGS.batch_size,
    shuffle=False,
    num_epochs=FLAGS.num_epochs_input)
 
  columns, targets = input_fn()
  features = columns['values']
  ids = columns['id']
  labels = targets
  return features, (ids, labels)

def data_provider_serving(preprocessing_fn, default_batch_size=None):
  feature_spec = {
    'indices' : tf.VarLenFeature(tf.int64),
    'values' : tf.VarLenFeature(tf.float32),
  }
  
  serialized_tf_example = tf.placeholder(dtype=dtypes.string,
                                                  shape=[default_batch_size],
                                                  name='input_example_tensor')
  tensors = {'examples': serialized_tf_example}
  examples = parsing_ops.parse_example(serialized_tf_example, feature_spec)
  indices = examples['indices']
  values = examples['values']

  indices_shape = tf.shape(indices.indices)
  rank = indices_shape[1]
  ids = tf.to_int64(indices.values)
  indices_columns_to_preserve = tf.slice(
        indices.indices, [0, 0], tf.stack([-1, rank - 1]))
  new_indices = tf.concat(
        [indices_columns_to_preserve, tf.reshape(ids, [-1, 1])], 1)
   
  new_shape = tf.concat([indices.dense_shape[:-1], [_NUM_DIMENSION]], 0)
  #[tf.cast(indices.dense_shape[0], tf.int64),  _NUM_DIMENSION]
  sp_tensor = tf.SparseTensor(new_indices, values.values, new_shape)
  features = tf.sparse_tensor_to_dense(sp_tensor, 0)

  return features, tensors

def data_provider_eval(dataset, preprocessing_fn):
  provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        num_readers=FLAGS.num_readers,
        num_epochs=FLAGS.num_epochs_input,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
  feature, label = provider.get(['feature', 'label'])
  #label -= FLAGS.labels_offset
  features, labels  = tf.train.batch(
        [feature, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)
  return features, labels

def data_provider_queue(dataset, preprocessing_fn):
  provider = slim.dataset_data_provider.DatasetDataProvider(
    dataset,
    num_readers=FLAGS.num_readers,
    num_epochs=FLAGS.num_epochs_input,
    common_queue_capacity=20 * FLAGS.batch_size,
    common_queue_min=10 * FLAGS.batch_size)
  feature, label = provider.get(['feature', 'label'])

  #label -= FLAGS.labels_offset
  features, labels  = tf.train.batch(
        [feature, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)
  #labels = slim.one_hot_encoding(
  #    labels, dataset.num_classes - FLAGS.labels_offset)
  batch_queue = slim.prefetch_queue.prefetch_queue(
        [features, labels], capacity=2 * FLAGS.num_clones)
 
  return batch_queue

####################
# Define the model #
####################
def clone_fn(batch_queue, network_fn, deploy_config):
  """Allows data parallelism by creating multiple clones of network_fn."""
  features, labels = batch_queue.dequeue()

  with slim.arg_scope([slim.model_variable, slim.variable],
      device=deploy_config.variables_device()):
    kwargs = {}
    kwargs['num_ps_replicas'] = deploy_config.num_ps_tasks
    #kwargs['num_ps_replicas'] = 2
    logits, end_points = network_fn(features, kwargs)
 
  #TODO: add metric/eval, thresholds=(.5)         
  #############################
  # Specify the loss function 
  ##############################
  loss = tf.losses.sigmoid_cross_entropy(labels, logits)
  return end_points

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'label': tf.FixedLenFeature([], tf.int64, default_value=0),
      'indices' : tf.VarLenFeature(tf.int64),
      'values' : tf.VarLenFeature(tf.float32),
  }

  items_to_handlers = {
      'label': slim.tfexample_decoder.Tensor('label'),
      'feature': slim.tfexample_decoder.SparseTensor('indices', 'values', 
              shape=[_NUM_DIMENSION], densify=True),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=None)
