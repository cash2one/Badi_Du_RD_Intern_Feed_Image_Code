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
"""
Contains the Tf kv record.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.contrib.slim.python.slim.data import data_decoder

from kvrecord import tf_kv_op 

class TfKvDecoder(data_decoder.DataDecoder):
  """
      pair example to tensor
  """

  def __init__(self, kv_path, is_training):
    """Constructs the decoder.

    Args:
     kv_path 
    """
    self._kv_table = tf_kv_op.tf_kv(
          shared_name="tf_kv_%s" % kv_path,
          key_dtype=dtypes.string,
          value_dtype=dtypes.string)

    filename = ops.convert_to_tensor(kv_path, dtypes.string)
    # pylint: disable=protected-access
    self._init_op = tf_kv_op.init_kv(
        self._kv_table,
        filename)
    # pylint: enable=protected-access
    ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, self._init_op)
    #ops.add_to_collection(ops.GraphKeys.ASSET_FILEPATHS, filename)
    self._default_value = ops.convert_to_tensor("", dtype=dtypes.string)
    self._is_training = is_training

  def init_kv(self):
    return self._init_op

  def list_items(self):
    """See base class, default name now"""
    return ["image_cur", "image_pos", "image_neg"] 
  
  def default_value(self):
    return self._default_value

  def decode(self, pair_example, items):
    """Decodes pair example
    Args: items should be equal to list_items now
    Returns:
      the decoded items, a list of tensor.
    """
    keys = tf.string_split([pair_example], delimiter='\t').values    
    if self._is_training:
      #image_url1\timage_url2\timage_url3
      values = tf_kv_op.lookup(self._kv_table, keys, self._default_value)
      outputs = []
      for i in range(len(items)): #len(items) equal to len(keys) 
        outputs.append(values[i])
    else:
      #{cur_nid}_{rec_nid} \t url1 \t url2 \t show \t click
      pair_id = tf.slice(keys, [0], [1])
      image_urls = tf.slice(keys, [1], [2])
      outputs = [pair_id]
      values = tf_kv_op.lookup(self._kv_table, image_urls, self._default_value)
      outputs.append(values[0])
      outputs.append(values[1])
    return outputs
