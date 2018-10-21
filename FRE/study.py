import tensorflow as tf
import numpy as np
import cv2
import time

im1 = cv2.imread('/home/liupengli/myWork/DataSets/HED-BSDS/train/aug_gt/0.0_1_0/2092.png')
im2 = cv2.imread('/home/liupengli/myWork/DataSets/HED-BSDS/train/aug_gt/0.0_1_0/8023.png')
im3 = cv2.imread('/home/liupengli/myWork/DataSets/HED-BSDS/train/aug_gt/0.0_1_0/3096.png')
im4 = cv2.imread('/home/liupengli/myWork/DataSets/HED-BSDS/train/aug_gt/0.0_1_0/8049.png')
im5 = cv2.imread('/home/liupengli/myWork/DataSets/HED-BSDS/train/aug_gt/0.0_1_0/8143.png')
im6 = cv2.imread('/home/liupengli/myWork/DataSets/HED-BSDS/train/aug_gt/0.0_1_0/12003.png')
im7 = cv2.imread('/home/liupengli/myWork/DataSets/HED-BSDS/train/aug_gt/0.0_1_0/12084.png')
im8 = cv2.imread('/home/liupengli/myWork/DataSets/HED-BSDS/train/aug_gt/0.0_1_0/12074.png')
# im = [im1, im2, im3, im4, im5, im6, im7, im8]
# im = im/255.0
# im[im > 0.5] = 1.0
# im[im <= 0.5] = 0.0
# im = tf.to_float(tf.stack(im))

# s1 = tf.reduce_sum(1.0 - im)
# s2 = tf.reduce_sum(im)
# count_neg = tf.reduce_sum((1.0 - im), axis=[1,2,3], keepdims=True) + 1e-5
# count_pos = tf.reduce_sum(im, axis=[1,2,3], keepdims=True) + 1e-5
# beta = tf.divide(count_neg, tf.add(count_neg, count_pos))
# sess = tf.InteractiveSession()
# print((s1-s2).eval())
# print(s2.eval())

# im_0 = tf.zeros_like(im)
# beta1 = tf.cast(tf.equal(im_0, im), tf.float32)
# beta1 = tf.reduce_mean(beta1, axis=[1,2,3], keepdims=True)
# print(beta1.eval())
# print((beta-beta1).eval())

a = tf.add_n([im1,im2,im3])