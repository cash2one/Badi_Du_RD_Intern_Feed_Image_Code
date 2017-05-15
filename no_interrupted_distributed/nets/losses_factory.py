#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
main for pairwise loss
File: nets/losses.py
Author: tuku(tuku@baidu.com)
Date: 2017/04/14 15:02:47
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def contrastive_loss(left, right, y, margin, extra=False, scope="constrastive_loss"):
    r"""Loss for Siamese networks as described in the paper:
    `Learning a Similarity Metric Discriminatively, with Application to Face
    Verification <http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf>`_ by Chopra et al.

    .. math::
        \frac{1}{2} [y \cdot d^2 + (1-y) \cdot \max(0, m - d)^2], d = \Vert l - r \Vert_2

    Args:
        left (tf.Tensor): left feature vectors of shape [Batch, N].
        right (tf.Tensor): right feature vectors of shape [Batch, N].
        y (tf.Tensor): binary labels of shape [Batch]. 1: similar, 0: not similar.
        margin (float): horizon for negative examples (y==0).
        extra (bool): also return distances for pos and neg.

    Returns:
        tf.Tensor: constrastive_loss (averaged over the batch), (and optionally average_pos_dist, average_neg_dist)
    """
    with tf.name_scope(scope):
        y = tf.cast(y, tf.float32)

        delta = tf.reduce_sum(tf.square(left - right), 1)
        delta_sqrt = tf.sqrt(delta + 1e-10)

        match_loss = delta
        missmatch_loss = tf.square(tf.nn.relu(margin - delta_sqrt))

        loss = tf.reduce_mean(0.5 * (y * match_loss + (1 - y) * missmatch_loss))

        if extra:
            num_pos = tf.count_nonzero(y)
            num_neg = tf.count_nonzero(1 - y)
            pos_dist = tf.where(tf.equal(num_pos, 0), 0.,
                                tf.reduce_sum(y * delta_sqrt) / tf.cast(num_pos, tf.float32),
                                name="pos-dist")
            neg_dist = tf.where(tf.equal(num_neg, 0), 0.,
                                tf.reduce_sum((1 - y) * delta_sqrt) / tf.cast(num_neg, tf.float32),
                                name="neg-dist")
            return loss, pos_dist, neg_dist
        else:
            return loss

def siamese_cosine_loss(left, right, y, scope="cosine_loss"):
    r"""Loss for Siamese networks (cosine version).
    Same as :func:`contrastive_loss` but with different similarity measurement.

    .. math::
        [\frac{l \cdot r}{\lVert l\rVert \lVert r\rVert} - (2y-1)]^2

    Args:
        left (tf.Tensor): left feature vectors of shape [Batch, N].
        right (tf.Tensor): right feature vectors of shape [Batch, N].
        y (tf.Tensor): binary labels of shape [Batch]. 1: similar, 0: not similar.

    Returns:
        tf.Tensor: cosine-loss as a scalar tensor.
    """

    def l2_norm(t, eps=1e-12):
        """
        Returns:
            tf.Tensor: norm of 2D input tensor on axis 1
        """
        with tf.name_scope("l2_norm"):
            return tf.sqrt(tf.reduce_sum(tf.square(t), 1) + eps)

    with tf.name_scope(scope):
        y = 2 * tf.cast(y, tf.float32) - 1
        pred = tf.reduce_sum(left * right, 1) / (l2_norm(left) * l2_norm(right) + 1e-10)

        return tf.nn.l2_loss(y - pred) / tf.cast(tf.shape(left)[0], tf.float32)


def triplet_loss(anchor, positive, negative, margin, extra=False, scope="triplet_loss"):
    r"""Loss for Triplet networks as described in the paper:
    `FaceNet: A Unified Embedding for Face Recognition and Clustering
    <https://arxiv.org/abs/1503.03832>`_
    by Schroff et al.

    Learn embeddings from an anchor point and a similar input (positive) as
    well as a not-similar input (negative).
    Intuitively, a matching pair (anchor, positive) should have a smaller relative distance
    than a non-matching pair (anchor, negative).

    .. math::
        \max(0, m + \Vert a-p\Vert^2 - \Vert a-n\Vert^2)

    Args:
        anchor (tf.Tensor): anchor feature vectors of shape [Batch, N].
        positive (tf.Tensor): features of positive match of the same shape.
        negative (tf.Tensor): features of negative match of the same shape.
        margin (float): horizon for negative examples
        extra (bool): also return distances for pos and neg.

    Returns:
        tf.Tensor: triplet-loss as scalar (and optionally average_pos_dist, average_neg_dist)
    """

    with tf.name_scope(scope):
        d_pos = tf.reduce_sum(tf.square(anchor - positive), 1)
        d_neg = tf.reduce_sum(tf.square(anchor - negative), 1)

        loss = tf.reduce_mean(tf.maximum(0., margin + d_pos - d_neg))

        if extra:
            pos_dist = tf.reduce_mean(tf.sqrt(d_pos + 1e-10), name='pos-dist')
            neg_dist = tf.reduce_mean(tf.sqrt(d_neg + 1e-10), name='neg-dist')
            return loss, pos_dist, neg_dist
        else:
            return loss

def soft_triplet_loss(anchor, positive, negative, extra=True, scope="soft_triplet_loss"):
    """Loss for triplet networks as described in the paper:
    `Deep Metric Learning using Triplet Network
    <https://arxiv.org/abs/1412.6622>`_ by Hoffer et al.

    It is a softmax loss using :math:`(anchor-positive)^2` and
    :math:`(anchor-negative)^2` as logits.

    Args:
        anchor (tf.Tensor): anchor feature vectors of shape [Batch, N].
        positive (tf.Tensor): features of positive match of the same shape.
        negative (tf.Tensor): features of negative match of the same shape.
        extra (bool): also return distances for pos and neg.

    Returns:
        tf.Tensor: triplet-loss as scalar (and optionally average_pos_dist, average_neg_dist)
    """

    eps = 1e-10
    with tf.name_scope(scope):
        d_pos = tf.sqrt(tf.reduce_sum(tf.square(anchor - positive), 1) + eps)
        d_neg = tf.sqrt(tf.reduce_sum(tf.square(anchor - negative), 1) + eps)

        logits = tf.stack([d_pos, d_neg], axis=1)
        ones = tf.ones_like(tf.squeeze(d_pos), dtype="int32")

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ones))

        if extra:
            pos_dist = tf.reduce_mean(d_pos, name='pos-dist')
            neg_dist = tf.reduce_mean(d_neg, name='neg-dist')
            return loss, pos_dist, neg_dist
        else:
            return loss

def cosine_distance(curr, pair):
    cosine = tf.div(tf.reduce_sum(tf.multiply(curr, pair), 1, keep_dims=True),\
        tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(curr), 1, keep_dims=True),\
        tf.reduce_sum(tf.square(pair), 1, keep_dims=True))))
    return cosine

#return a int tensor,means avg of a batch loss
def triplet_cosine_loss(current, positive, negative, margin, extra=False, scope="triplet_cosine_loss"):
    #return shape [batch_size, 1]
    with tf.name_scope(scope):
      pos_cosine = cosine_distance(current, positive)
      neg_cosine = cosine_distance(current, negative)
      pair_losses = tf.reduce_mean(tf.maximum(0., margin - (pos_cosine - neg_cosine))) 
      if extra:
        pos_dist = tf.reduce_mean(pos_cosine + 1e-10, name='pos-dist')
        neg_dist = tf.reduce_mean(neg_cosine + 1e-10, name='neg-dist')
        return pair_losses, pos_dist, neg_dist
      else:
        return pair_losses


