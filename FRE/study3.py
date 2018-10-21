import tensorflow as tf
import numpy as np
import cv2
import os

img1 = cv2.imread('/home/liupengli/Desktop/3063.png')
cv2.imshow('win1', img1)
# cv2.waitKey()
img2 = cv2.imread('/home/liupengli/Desktop/5096.png')
img = tf.cast(tf.stack([img1, img2]), tf.float32)
a = tf.divide(img, tf.constant(255.0))



condition1 = a <= 0.5
condition2 = a > 0.0
condition = tf.logical_and(condition1, condition2)
c = tf.where(condition, tf.zeros(shape=a.shape), tf.ones(shape=a.shape))
c = c * a
with tf.Session() as sess:
	np = sess.run(c)
	cv2.imshow('win', np[0])
	cv2.waitKey()