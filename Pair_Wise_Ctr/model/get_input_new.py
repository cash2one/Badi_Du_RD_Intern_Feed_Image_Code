#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Pw @ 2017-03-09 10:58:12
import sys
import tensorflow as tf
import time
import os

def get_input(data_path, batch_size):
    w = 218
    h = 146
    c = 3
    rw = 200
    rh = 126
    data_files = tf.gfile.Glob(data_path)
    #filename_queue = tf.train.string_input_producer(data_files, shuffle=True, num_epochs=2)
    filename_queue = tf.train.string_input_producer(data_files, shuffle=True)

    reader = tf.TextLineReader()
    key, val = reader.read(filename_queue)

    #load string from file,need split,and convert
    sv = tf.string_split([val], '\t').values
    key_img_b64 = sv[0]
    clicked_img_b64 = sv[1]
    not_clicked_img_b64 = sv[2]

    key_img_str = tf.decode_base64(key_img_b64)
    clicked_img_str = tf.decode_base64(clicked_img_b64)
    not_clicked_img_str = tf.decode_base64(not_clicked_img_b64)

    #decode to [h, w, c],before enqueue example,each of image should reshape to fixed shapes
    key_img_tensor = tf.image.resize_image_with_crop_or_pad(tf.image.decode_image(key_img_str, channels=c), rh, rw)
    key_img_tensor = tf.reshape(key_img_tensor, [rh, rw, c])
    clicked_img_tensor = tf.image.resize_image_with_crop_or_pad(tf.image.decode_image(clicked_img_str, channels=c), rh, rw)
    clicked_img_tensor = tf.reshape(clicked_img_tensor, [rh, rw, c])
    not_clicked_img_tensor = tf.image.resize_image_with_crop_or_pad(tf.image.decode_image(not_clicked_img_str, channels=c), rh, rw)
    not_clicked_img_tensor = tf.reshape(not_clicked_img_tensor, [rh, rw, c])

    key_img_tensor = tf.image.random_flip_left_right(key_img_tensor)
    clicked_img_tensor = tf.image.random_flip_left_right(clicked_img_tensor)
    not_clicked_img_tensor = tf.image.random_flip_left_right(not_clicked_img_tensor)

    key_img_tensor = tf.image.per_image_standardization(key_img_tensor)
    clicked_img_tensor = tf.image.per_image_standardization(clicked_img_tensor)
    not_clicked_img_tensor = tf.image.per_image_standardization(not_clicked_img_tensor)

    example_queue = tf.FIFOQueue(
            capacity=16 * batch_size * 3,
            dtypes=[tf.float32],
            shapes=[[rh, rw, c]])
    num_threads = 16
    #enqueue, muti threads
    enqueue_op = example_queue.enqueue_many([[key_img_tensor, clicked_img_tensor, not_clicked_img_tensor]])
    tf.train.add_queue_runner(tf.train.QueueRunner(example_queue, 
        [enqueue_op] * num_threads))
   
    #dequeue batch of example
    #key_images, clicked_imgs, not_clicked_imgs = example_queue.dequeue_many(batch_size)
    #return key_images, clicked_imgs, not_clicked_imgs
    sample_imgs = example_queue.dequeue_many(batch_size * 3)
    return sample_imgs

    #with tf.Session() as sess:
    #    tf.train.start_queue_runners()
    #    print sess.run(tf.shape(sample_imgs))
    #    print sess.run(sample_imgs)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    data_path = sys.argv[1]
    batch_size = int(sys.argv[2])
    get_input(data_path, batch_size)
    '''
    #train model
    key_images, clicked_images, not_clicked_images = get_input(data_path, batch_size)

    num_d = 218 * 146 * 3
    num_class = 2
    #[batch_size h w c] to [batch_size h*w*c]
    key_images = tf.reshape(key_images, [batch_size, num_d])
    clicked_images = tf.reshape(clicked_images, [batch_size, num_d])
    not_clicked_images = tf.reshape(not_clicked_images, [batch_size, num_d])

    with tf.device('/gpu:1'):
        w = tf.random_normal([num_d, num_class])
        y1 = tf.matmul(key_images, w)
        y2 = tf.matmul(clicked_images, w)
        y3 = tf.matmul(not_clicked_images, w)
    
    init = tf.initialize_all_variables()

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.device('/gpu:3'):
            sess.run(init)
            tf.train.start_queue_runners()
            t1 = time.time()
            print "batch_num: " + str(len(sess.run(key_images)))
            for i in range(1000):
            #sess.run(train_step, feed_dict={x: images, y_: labels})
                print "iter: " + str(i)
                #print sess.run([y1, y2, y3])
                print sess.run([y1, y2, y3])
                t2 = time.time()
                print "used time: " + str(t2 - t1)
                #print t2 - t1
                t1 = t2
'''    
