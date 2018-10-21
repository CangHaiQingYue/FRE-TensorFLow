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

def convert_to_example(writer, image_path, label_path):
    
    # check if image with the right relative GT
    if image_path[-9:-4] != label_path[-9:-4]:
        raise NameError
    img = cv2.imread(image_path)   
    if len(img.shape) < 2:
        raise ValueError

    h, w, c = img.shape

    image = tf.gfile.GFile(image_path, 'rb').read()
    label = tf.gfile.GFile(label_path, 'rb').read()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(image),
        'label': _bytes_feature(label),
        'h': _int64_feature(h),
        'w': _int64_feature(w),
        'c': _int64_feature(c)
        }))
    writer.write(example.SerializeToString())


# path = '/home/liupengli/myWork/DataSets/HED-BSDS/train'
# augPath = ['aug_data', 'aug_data_scale_0.5', 'aug_data_scale_1.5']
# gtPath = ['aug_gt', 'aug_gt_scale_0.5', 'aug_gt_scale_1.5']

# imagePath = sorted(os.listdir(os.path.join(path, augPath[0])))                   #32 floders
# allImageName = sorted(os.listdir(os.path.join(path, augPath[1], imagePath[0])))  #300  jpg name 
# allLabelName = sorted(os.listdir(os.path.join(path, gtPath[1], imagePath[0])))   #300  png name
# # print(allLabelName)

base_path = '/home/liupengli/myWork/FRE/image'
img = ['img']
lab = ['lab']

allImageName = sorted(os.listdir(os.path.join(base_path, img[0])))
allLabelName = sorted(os.listdir(os.path.join(base_path, lab[0])))

with tf.python_io.TFRecordWriter('/home/liupengli/myWork/FRE/FRE/tfrecord/data.tfrecord') as writer:
    for idx1, fileName in enumerate(zip(allImageName, allLabelName)):
        image_path = os.path.join(base_path, img[0], fileName[0])
        label_path = os.path.join(base_path, lab[0], fileName[1])
        print(image_path)
        print(label_path)
        convert_to_example(writer, image_path, label_path)


