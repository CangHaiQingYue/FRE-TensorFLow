import tensorflow as tf 
import numpy as np
import cv2
import os
# from model import fre_model
from model import try_model_1 as fre_model


mean_pixel_value = [104.00699, 116.66877, 122.67892]
tf.app.flags.DEFINE_string('checkpoint_path', '/home/liupengli/myWork/FRE/checkpoints/fre.ckpt-42000',
                            'where the .cpkt file is')                       
tf.app.flags.DEFINE_string('test_list', '/home/liupengli/myWork/DataSets/HED-BSDS/test2.lst', 
                            'paht, where the test.lst is')
                           
tf.app.flags.DEFINE_string('image_path', '/home/liupengli/myWork/DataSets/HED-BSDS/test',
                            'image path, except image name')

tf.app.flags.DEFINE_string('model_name', 'FRE',
                             'model_name')
tf.app.flags.DEFINE_string('run_state', 'testing',
                             'wheter training or testing, used for batch_norm')
FLAGS = tf.app.flags.FLAGS


def read_image(image_name):
    '''
    1: thorugh this way, images are not in RGB order, so we should transpose the channel
    2. we should substract the mean_value of each channel for pre-processing
    3. cast image dtype to float also is very important
    '''

    image = tf.read_file(image_name)
    image = tf.image.decode_image(image)
    #    !!!!!!!  DONOT USE  tf.image.convert_image_dtype !!!!!!!!
    if image.dtype != tf.float32: 
        image = tf.cast(image, tf.float32)
    image = tf.reshape(image, shape=[321,481, 3])
    image = tf.image.resize_images(image, [320,480])
    image = transpose_image(image)
    image = image - tf.convert_to_tensor(mean_pixel_value)
    return image 
def transpose_image(img):
    R = tf.slice(img, begin=[0,0,0], size=[320,480,1])
    G = tf.slice(img, begin=[0,0,1], size=[320,480,1])
    B = tf.slice(img, begin=[0,0,2], size=[320,480,1])
    O = tf.stack([B, G, R], axis=3)
    O = tf.reshape(O, shape=[320,480,3])
    return O


def main(_):
   
    #get testing images
    #
    filename = []
    path = tf.gfile.Open(FLAGS.test_list)
    readlines = path.readlines()                                          # just for geting
    readline = [re.strip('\n') for re in readlines]                #  <----
    for i in range(len(readline)):                                        # all images path
        filename.append(os.path.join(FLAGS.image_path, readline[i][5:]))  
    path.close()
    # print(filename)
    dataset = tf.data.Dataset.from_tensor_slices(filename)
    dataset = dataset.map(read_image)
    dataset = dataset.batch(1)
    iterator = dataset.make_one_shot_iterator()
    image = iterator.get_next()
    # print(dataset.output_shapes)

    # init model
    net_init = fre_model.FRE(FLAGS.run_state)
    end_points = net_init.net(FLAGS.model_name, image)
   
    # pre-processing
    side_outputs = []
    for side_output in end_points:
        side_outputs.append(tf.nn.sigmoid(side_output))
        # side_outputs.append(side_output)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        for _ in range(1):
            saver.restore(sess, FLAGS.checkpoint_path)
            edges =  sess.run(side_outputs)
            print('np_max=', np.max(edges))
            edgemaps = [edge[0] for edge in edges]

            for idx, em in enumerate(edgemaps):
                em = 255.0 * em
                h, w, c = np.shape(em)
                if w > h:
                    em = cv2.resize(em,(481,321))  #acorss
                    # em = cv2.resize(em,(560,425))  #acorss
                else:
                    em = cv2.resize(em,(321,481))    #vertical 
                    # em = cv2.resize(em,(425,526))    #vertical
                if idx != 5:
                    cv2.imshow('win', np.uint8(em))
                    cv2.waitKey()
                    pass
                    # cv2.imwrite(os.path.join(self.cfgs['test_output'], 'testing-{}-{:03}.png'.format(index, idx)), np.uint8(em))
                else:
                    # em.save(os.path.join(self.cfgs['test_output'], 'testing-{}-{:03}.png'.format(img[5:-4])), em)
                    # cv2.imwrite(os.path.join(self.cfgs['test_output'], '{}.png'.format(img[5:-4])), np.uint8(em))
                    cv2.imwrite('/home/liupengli/myWork/FRE/image/haha.png', np.uint8(em))
                    cv2.imshow('win', np.uint8(em))
                    cv2.waitKey()









if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()