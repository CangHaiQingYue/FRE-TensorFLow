import tensorflow as tf 
import random
import cv2
import os

_SAMEPLE_PER_RECORD = 4900


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_example(writer, image_path, label_path):
    
    # check if image with the right relative GT
    if image_path[-9:-4] != label_path[-9:-4]:
        raise NameError

    img = cv2.imread(image_path) 
    if len(img.shape)< 2:
        raise ValueError
    h, w, c = img.shape
    
    image = tf.gfile.FastGFile(image_path, 'rb').read()
    label = tf.gfile.FastGFile(label_path, 'rb').read()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(image),
        'label': _bytes_feature(label),
        'h': _int64_feature(h),
        'w': _int64_feature(w),
        'c': _int64_feature(c)
        }))
    writer.write(example.SerializeToString())



# name = '/home/liupengli/myWork/DataSets/HED-BSDS/train_pair.lst'
# base_path = '/home/liupengli/myWork/DataSets/HED-BSDS'
name = '/home/liupengli/myWork/DataSets/rcf/bsds_pascal_train_pair.lst'
base_path = '/home/liupengli/myWork/DataSets/rcf'
with tf.gfile.GFile(name) as f:
    filename = f.readlines()
    f.close()
filenames = [c.split(' ')for c in filename]
random.shuffle(filenames) 
print(len(filename[0]))
print(len(filenames))


num_record = int(len(filenames) / _SAMEPLE_PER_RECORD)
print('There are {} tfrecord files in all'.format(num_record))
k = 0
for i in range(num_record):
    record_name = '/home/liupengli/myWork/FRE/FRE/tfrecord/BSDS{num}.tfrecord'.format(num=i)
    with tf.python_io.TFRecordWriter(record_name) as writer:
        j = 0
        while j < _SAMEPLE_PER_RECORD and k < len(filenames):    
            image_path = (os.path.join(base_path, filenames[k][0])).strip()
            label_path = (os.path.join(base_path, filenames[k][1])).strip()
            convert_to_example(writer, image_path, label_path)
            # print(image_path)
            print('image in {img} and {gt} convert done!'.format(img=filenames[k][0], gt=filenames[k][1]))
            j += 1
            k += 1 
           


        
