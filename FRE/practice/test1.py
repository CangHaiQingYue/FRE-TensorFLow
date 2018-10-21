# This file coverts jpeg or png images to tfrecord
#
#==============================================================

import os
import tensorflow as tf
import cv2
import numpy as np


def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def convert_to_example(writer, image_path, label_path=None):
    
    image = tf.gfile.FastGFile(image_path, 'rb').read()
    label = tf.gfile.FastGFile(label_path, 'rb').read()
    # image = cv2.imread(image_path)
    # image = cv2.resize(image,(320,320))
    # label = cv2.imread(label_path)
    # label = cv2.resize(label,(320,320))

    # image = image.tobytes()
    # label = label.tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(image),
        'label': _bytes_feature(label)
        }))
    writer.write(example.SerializeToString())




imageName = '/home/liupengli/myWork/DataSets/HED-BSDS/train/aug_data/0.0_1_0/2092.jpg'
labelName = '/home/liupengli/myWork/DataSets/HED-BSDS/train/aug_gt/0.0_1_0/2092.png'
													
# if imageName[-5]
writer = tf.python_io.TFRecordWriter(
            '/home/liupengli/myWork/FRE/FRE/tfrecord/data.tfrecord')

convert_to_example(writer, imageName, labelName)
           
        # print(k)
# print('converting images in {filename} and {filename1} done!'
#        .format(filename=fileName[0], filename1=fileName[1]))

writer.close()