import tensorflow as tf 
import numpy as np
import time
import cv2
import os
from model import fre_model
# from model import try_model_1 as fre_model
# from model import new_model as fre_model


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
    image = cv2.imread(image_name)
    h, w, c = image.shape
    image = np.float32(image)
    image -= mean_pixel_value
    image = cv2.resize(image, (w-1, h-1))
    return image 

def main(_):
  
    #get testing images
    filename = []
    path = tf.gfile.Open(FLAGS.test_list)
    readlines = path.readlines()                                          # just for geting
    readline = [re.strip('\n') for re in readlines]                #  <----
    for i in range(len(readline)):                                        # all images path
        filename.append(os.path.join(FLAGS.image_path, readline[i][5:]))  
    path.close()
    # print(filename)
    image = tf.placeholder(dtype=tf.float32,shape=[None, None, None, 3]) 
 
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
        saver.restore(sess, FLAGS.checkpoint_path)
        start = time.time()
        for i in range(len(filename)):
            img = read_image(filename[i])
            edges =  sess.run(side_outputs, feed_dict={image: [img]})
            # print('np_max=', np.max(edges))
            edgemaps = [edge[0] for edge in edges]

            for idx, em in enumerate(edgemaps):
                em = 255.0 * em
                h, w, c = np.shape(em)
                if w > h:
                    em = cv2.resize(em,(481,321))                     
                else:
                    em = cv2.resize(em,(321,481))                     
                if idx != 5:
                    # cv2.imshow('win', np.uint8(em))
                    # cv2.waitKey()
                    pass
                    # cv2.imwrite(os.path.join(self.cfgs['test_output'], 'testing-{}-{:03}.png'.format(index, idx)), np.uint8(em))
                else:
                    cv2.imwrite('/home/liupengli/myWork/FRE/image/{}.png'.format(filename[i][46:-4]), np.uint8(em))
                    print('/home/liupengli/myWork/FRE/image/{}.png'.format(filename[i][46:-4]))
                    # print(filename[i])
                    # cv2.imshow('win', np.uint8(em))
                    # cv2.waitKey()
        fps = len(filename) / (time.time() - start)
        print('Testing time is {} s'.format(time.time() - start))
        print('FPS = ', fps)



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()