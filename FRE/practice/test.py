# To generate tf.dataset for network
#
#==============================================================

import tensorflow as tf
import numpy as np
import cv2
import os

mean_pixel_value = [104.00699, 116.66877, 122.67892]
def decode(image):
    return tf.image.decode_image(image)
    # return tf.decode_raw(image, tf.uint8)
def _parse_function(serialized_example):
    feature = {'image': tf.FixedLenFeature((), tf.string, default_value=''),
         'label': tf.FixedLenFeature((), tf.string, default_value='')
       }
    parse_feature = tf.parse_single_example(serialized_example, feature)
    # image = tf.image.decode_jpeg(parse_feature['image'])
    image = decode(parse_feature['image'])
    label = decode(parse_feature['label'])
    image = tf.reshape(image, shape=[ -1,544, 3])
    # image = transpose_image(image)
    label = tf.reshape(label, shape=[ 384,544, -1])
    return image, label

def transpose_image(img):
    R = tf.slice(img, begin=[0,0,0], size=[320,320,1])
    G = tf.slice(img, begin=[0,0,1], size=[320,320,1])
    B = tf.slice(img, begin=[0,0,2], size=[320,320,1])
    O = tf.stack([B, G, R], axis=3)
    O = tf.reshape(O, shape=[320,320,3])
    return O
def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image = image - tf.convert_to_tensor(mean_pixel_value)
    label = tf.cast(label, tf.float32) / 255.0
    return image, label

def get_iterator(mini_batch, path):
    path = '/home/liupengli/myWork/FRE/FRE/tfrecord/data.tfrecord'
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(_parse_function)
    # dataset = dataset.map(normalize)
    dataset = dataset.shuffle(buffer_size=600)
    # dataset = dataset.batch(mini_batch)
    dataset = dataset.repeat()
    # print(dataset.output_shapes)
    iterator = dataset.make_one_shot_iterator()
    image, label = iterator.get_next()
    # print(dataset.output_shapes)
    return image, label


def get_data(flags):
    record_path = flags.tfrecord_path
    recordPath = os.listdir(record_path)
    path = []
    for i in recordPath:
        path.append(os.path.join(record_path,i))
    image, label = get_iterator(flags.mini_batch, path)
    return image, label




if __name__ == '__main__':
    record_path = '/home/liupengli/myWork/FRE/FRE/tfrecord'
    recordPath = os.listdir('/home/liupengli/myWork/FRE/FRE/tfrecord')

    path = []
    for i in recordPath:
        path.append(os.path.join(record_path,i))
    # print(path)
    image, label = get_iterator(1,path)
    print('label.shape', label.shape)
    with tf.Session() as sess:
        for _ in range(1):
            img = sess.run(label)
            print(np.shape(img))
            cv2.imshow('win', img)
            cv2.waitKey()
            print(np.max(img))
