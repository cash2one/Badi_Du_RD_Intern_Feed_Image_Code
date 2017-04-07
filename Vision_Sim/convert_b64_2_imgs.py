#!/usr/bin/env python
########################################################################
# 
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
'''
File: convert_b64_2_imgs.py
Author: feedrd(feedrd@baidu.com)
Date: 2017/04/06 14:33:15
'''

import os 
import sys
import base64
from PIL import Image

all_imgs_name = {}
def convert(input_file_path, output_file_path):
    target_file = open(input_file_path, 'r')
    for line in target_file:
        cols = line.strip().split('\t')
        nid = cols[0]
        #type = cols[1]
        b64_str = cols[4]
        #img_name = nid+'.'+type
        #if nid not in all_imgs_name:
        #    all_imgs_name[nid] = img_name
        img_data = base64.b64decode(b64_str)
        cache_file =open('./tmp', 'wb')
        cache_file.write(img_data)
        cache_file.close()

        img = Image.open('./tmp')
        type = img.format
        if img.mode not in ('RGB'):
            continue
        img = img.resize((227,227),Image.ANTIALIAS)
        #type = img.format

        img_name = nid+'.'+type
        if nid not in all_imgs_name:
            all_imgs_name[nid] = img_name

        final_file_path = os.path.join(output_file_path, img_name)
        img.save(final_file_path)


def gen_vision_negative_samples(input_file_path, output_file_path, imgs_file_path):
    target_file = open(input_file_path, 'r')
    result_file = open(output_file_path, 'w+')

    for line in target_file:
        cols = line.strip().split('\t')
        sample1_nid = cols[1]
        sample2_nid = cols[2]
        if sample1_nid in all_imgs_name and sample2_nid in all_imgs_name:
            sample1_name = all_imgs_name[sample1_nid]
            sample2_name = all_imgs_name[sample2_nid]
            
            sample1_path = os.path.join(imgs_file_path, sample1_name)
            sample2_path = os.path.join(imgs_file_path, sample2_name)

            sample1 = open(sample1_path, 'rb')
            sample2 = open(sample2_path, 'rb')

            sample1_b64 = base64.urlsafe_b64encode(sample1.read())
            sample2_b64 = base64.urlsafe_b64encode(sample2.read())
            
            result_file.write('%d\t%s\t%s\n' % (0, sample1_b64, sample2_b64))
            
            sample1.close()
            sample2.close()

    target_file.close()
    result_file.flush()
    result_file.close()



if __name__ == '__main__':

    target_file_path = '/home/feedrd/leijinyi/feed_image/dataset/negative_data/match_images/'
    imgs_path = '/home/feedrd/leijinyi/feed_image/dataset/negative_data/imgs/'
    
    negative_samples_nid_file_path = '/home/feedrd/leijinyi/feed_image/dataset/negative_data/negative_samples_file'
    negative_samples_b64_file_path = '/home/feedrd/leijinyi/feed_image/dataset/negative_data/negative_samples_b64'
    file = os.listdir(target_file_path)
    for item in file:
        absolute_path = os.path.join(target_file_path, item)
        convert(absolute_path, imgs_path) 
    
    gen_vision_negative_samples(negative_samples_nid_file_path, negative_samples_b64_file_path, imgs_path)
