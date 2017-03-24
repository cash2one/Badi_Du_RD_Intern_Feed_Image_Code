#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Pw @ 2017-03-07 17:21:40

import urllib
import urllib2
import os
import sys
import time

img_path='./databases'

def download(img_path,url,index):
    if '.jpg' in url:
        try:
            html_data=urllib2.urlopen(url,None,5)
            img=html_data.read()
        except Exception:
            pass
        else:
            filename=os.path.join(img_path,str(index)+'.jpg')
            with open(filename,'wb') as f:
                f.write(img)

'''
def download(img_path,url,index):
    if '.jpg' in url:
        format='.jpg'
        tmp_original_file=img_path+'/'+str(index)+format
        
        cmd_line = 'wget -t 2 -T 20 --referer="www.baidu.com" "' \
                + url + '" -O ' \
                + tmp_original_file
'''
'''    
    #cmd_line='wget -t 2 -T 10 --referer="www.baidu.com" "'+url+'" '+tmp_original_file
        ret = os.system(cmd_line)
        if ret!=0:
            ret = os.system(cmd_line)
            if ret!=0:
                return 
            else:
                return 
        else:
            return 
'''

counter=0
for line in sys.stdin:
    line=line.strip()
    download(img_path,line.split('\t')[1].strip(),counter)
    #flag=download(img_path,line.split('\t')[1].strip())
    '''
    if flag:
        counter=counter+1
    '''
    counter=counter+1
#print counter
    #counter=counter+1
