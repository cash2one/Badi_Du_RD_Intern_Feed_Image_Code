import tensorflow as tf
import tensorflow.python.platform
import os
import sys
import base64
from PIL import Image

database='./databases'
files=os.listdir(database)
filenames=[]
for each in files:
    fullname=os.path.join(database,each)
    filenames.append(fullname)
#print filenames[0]
def Record_2_file(file,filenames):
    writer=tf.python_io.TFRecordWriter(file)
    index=0
    for each in filenames:
    #if index<=400:
        img=Image.open(each)
    #img=open(each,'rb')
    #img_raw=img.read()
    #img_encode=base64.b64encode(img_raw)
        if img.mode not in ('RGB'):
            continue
        #width, heigh=img.size
        #if width>227 and heigh>227:
        img=img.resize((75,75),Image.ANTIALIAS)
        img_raw=img.tobytes()
        example=tf.train.Example(features=tf.train.Features(feature={\
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),\
                #'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),\
                #'heigh':tf.train.Feature(int64_list=tf.train.Int64List(value=[heigh])),\
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))\
                }))
        writer.write(example.SerializeToString())
        index=index+1
        img.close()
        #index=index+1
    writer.close()
    print index

for i in range(100):
    file="../image/encode/img_data"+str(i)+".tfrecords"
    Record_2_file(file,filenames)

#print index
print "done"
#print filenames[0]

#filename_queue=tf.train.string_input_producer(filenames)

#print filename_queue
'''
reader=tf.TextLineReader()
key,value=reader.read(filename_queue)
record_defaults='test'
col0=tf.decode_csv(value,record_defaults=record_defaults)

#col0=tf.decode_csv(value)
with tf.Session() as sess:
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord)
    for i in range(1000):
        print sess.run([col0])

    coord.request_stop()
    coord.join(threads)
'''


