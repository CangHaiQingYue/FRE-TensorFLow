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
         'label': tf.FixedLenFeature((), tf.string, default_value=''),
         'h': tf.FixedLenFeature((),tf.int64),
         'w': tf.FixedLenFeature((),tf.int64),
         'c': tf.FixedLenFeature((),tf.int64)
         # 'c_lab': tf.FixedLenFeature((),tf.int64)
       }
    parse_feature = tf.parse_single_example(serialized_example, feature)

    image = tf.image.decode_image(parse_feature['image'])
    label = tf.image.decode_png(parse_feature['label'], channels=1)

    h = tf.cast(parse_feature['h'], tf.int64)
    w = tf.cast(parse_feature['w'], tf.int64)
    c_img = tf.cast(parse_feature['c'], tf.int64)

    image_shape = tf.stack([h, w, c_img])
    image = tf.reshape(image, image_shape)
    label = tf.reshape(label, tf.stack([h,w,1]))

    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)

    image = tf.image.resize_images(image, size=(320,320))
    label = tf.image.resize_images(label, (320,320))
    image = transpose_image(image)

    dim = label.get_shape().as_list()[-1]
 
    print('label.shape', label.shape)
    
    if dim != 1:
        print('Master: The Dimension of groundTrth must =1 !')
        raise ValueError
    # image = tf.reshape(image, shape=[544,384,3])
    # image = tf.reshape(image, shape=[320, 320, 3])
    # label = tf.reshape(label, shape=[320, 320, -1])
    return image, label

def transpose_image(img):
    if 320 in img.get_shape().as_list():
        R = tf.slice(img, begin=[0,0,0], size=[320,320,1])
        G = tf.slice(img, begin=[0,0,1], size=[320,320,1])
        B = tf.slice(img, begin=[0,0,2], size=[320,320,1])
        O = tf.stack([B, G, R], axis=3)
        O = tf.reshape(O, shape=[320,320,3])
        return O
    else:
        print("Master: all images should be resized to 320x320!!!")
        raise ValueError

def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image = image - tf.convert_to_tensor(mean_pixel_value)
    label = tf.cast(label, tf.float32) / 255.0
    return image, label

def get_iterator(mini_batch, path):
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(_parse_function)
    dataset = dataset.map(normalize)
    # dataset = dataset.shuffle(buffer_size=600)
    # dataset = dataset.batch(mini_batch)
    dataset = dataset.repeat()
    # print(dataset.output_shapes)
    iterator = dataset.make_one_shot_iterator()
    image, label  = iterator.get_next()
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
    base_path = '/home/liupengli/myWork/FRE/FRE/tfrecord'
    recordPath = os.listdir(base_path)

    path = []
    for i in recordPath:
        path.append(os.path.join(base_path,i))
    # print(path)
    image, label = get_iterator(1,path)
    # image = tf.image.resize_images(image, [320,320])
    with tf.Session() as sess:
        for _ in range(8):
            img = sess.run(image)
            print(np.shape(img))
            cv2.imshow('win', np.uint8(img))
            cv2.waitKey()
            # print('high=',high.eval())
