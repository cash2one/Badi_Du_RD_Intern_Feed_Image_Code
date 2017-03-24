#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Pw @ 2017-03-07 19:47:35
import time
import sys
import tensorflow.python.platform
import tensorflow as tf
import os 
import base64
#from PIL import Image


def read_and_decode(filename_list):
    filename_queue = tf.train.string_input_producer(filename_list,num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    feature = tf.parse_single_example(serialized_example,\
            features={'label': tf.FixedLenFeature([], tf.int64),\
            #'width':tf.FixedLenFeature([1], tf.int64),\
            #'heigh':tf.FixedLenFeature([1], tf.int64),\
            'img_raw' : tf.FixedLenFeature([], tf.string),\
            })
    #print features['img_raw']
    #features['img_raw']=base64.b64decode(features['img_raw'])
    #width=tf.cast(features['width'], tf.int64)
    #heigh=tf.cast(features['heigh'], tf.int64)
    img = tf.decode_raw(feature['img_raw'], tf.uint8)
    #print img.shape
    #img=base64.b64decode(img)
    img = tf.reshape(img, [227, 227, 3])
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    float_img=tf.cast(img, tf.float32)
    norm_img = tf.image.per_image_standardization(float_img)

    label = tf.cast(feature['label'], tf.int64)
    return norm_img, label


database='./encode'
files=os.listdir(database)
filenames=[]
for each in files:
    fullname=os.path.join(database,each)
    filenames.append(fullname)

img, label = read_and_decode(filenames)

img_batch, label_batch = tf.train.shuffle_batch([img, label],\
        batch_size=128, capacity=640,\
        num_threads=2,\
        min_after_dequeue=256)
label_batch=tf.reshape(label_batch,[128])
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    counter=0
    start_time = time.time()
    for i in range(40):
        
        #if coord.should_stop():
        #    print counter
        #    counter=counter+1
        #    continue

        val, l= sess.run([img_batch, label_batch])
        #img=val[0]
        #save_img=Image.fromarray(img,'RGB')
        #save_img.save('./'+str(i)+'.jpg')
        #counter=counter+1
        print (val.shape,l)
        '''
        tf.errors.OutOfRangeError:
            print "done"
        else:
            print(val.shape, l)
        finally:
        '''
    duration = time.time() - start_time

    coord.request_stop()
    coord.join(threads)
    print '%f' % (duration/(40*128))

    
