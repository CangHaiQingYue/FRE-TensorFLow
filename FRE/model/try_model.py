from tensorflow.python.training.moving_averages import assign_moving_average
import tensorflow as tf
from model import deconv
from model.loss import fre_loss
slim = tf.contrib.slim

class FRE():
    def __init__(self, run_state):
        if run_state == 'training':
            self.model_state = True
        else:
            self.model_state = None
    def net(self, model_name, inputs):
        return fre_net(model_name, inputs,self.model_state)
    def losses(self, end_points, labels):
        return fre_losses(end_points, labels)




def fre_net(model_name, inputs, model_state):
    with tf.variable_scope(model_name,'FRE', [inputs],
        regularizer=tf.contrib.layers.l2_regularizer(0.0002)):
        end_points = {}
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['stage1'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['stage2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['stage3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        end_points['stage4'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['stage5'] = net
        tf.logging.info('Backbone Configure Done!')
        
        #  get side_output layer.  For block_1, just performing mapping without deconvolution
        with tf.variable_scope('freBlock_1'):
            freBlock_1 = conv_layer(end_points['stage1'], [1,1,64,1], b_shape=1,
                            w_init= tf.truncated_normal_initializer(stddev=0.01),
                            b_init=tf.constant_initializer(0),
                            activation=False,
                            name='side_1')

        freBlock_2 = side_layer(end_points['stage2'], "side_2", 2, model_state)  
        freBlock_3 = side_layer(end_points['stage3'], "side_3", 4, model_state)
        freBlock_4 = side_layer(end_points['stage4'], "side_4", 8, model_state)
        freBlock_5 = side_layer(end_points['stage5'], "side_5", 16,model_state)
    
        
        
        fuse_1 = [freBlock_1, freBlock_2, freBlock_3, freBlock_4, freBlock_5]

        w_shape = [1, 1, len(fuse_1), 1]
        fuse = conv_layer(tf.concat(fuse_1, axis=3),
                                    w_shape, b_shape=1,
                                    w_init=tf.truncated_normal_initializer(stddev=0.01),
                                    b_init=tf.constant_initializer(0),
                                    name = 'fuse')

      
        side_output = [freBlock_1, freBlock_2, freBlock_3, freBlock_4,
                                freBlock_5, fuse]
        tf.logging.info('Side_output Configure Done!')
        return side_output

def fre_losses(end_points, labels):
    with tf.variable_scope('loss'):
        loss = 0.0
        for _, key in enumerate(end_points):
            cost = fre_loss(key, labels)
            loss += cost
    tf.losses.add_loss(loss)
    
    if loss is not None:
        tf.logging.info('Add_loss Done!')
        tf.summary.scalar('weights_loss', loss)

def side_layer(inputs, name, upscale, model_state):
    with tf.variable_scope(name):
        in_shape = inputs.shape.as_list()
        w_shape = [1, 1, in_shape[-1], 32]

        if upscale == 16:
            rate = 10
        elif upscale == 8:
            rate = 8
        elif upscale == 4:
            rate = 6
        else:
            rate = 4
        classifier = inputs
        short = conv_layer(classifier, [1,1,in_shape[-1],128], model_state,
                                     use_bias=False,
                                     w_init=tf.contrib.layers.xavier_initializer(),
                                     activation=True,
                                     rate=1,
                                     name=name + '_short')
        classifier = conv_layer(classifier, w_shape, model_state,
                                     use_bias=False,
                                     w_init=tf.contrib.layers.xavier_initializer(),
                                     activation=True,
                                     rate=1,
                                     name=name + '_reduction')

        classifier = conv_layer(classifier, [3,3,32,32], model_state,
                                     use_bias=False,
                                     w_init=tf.contrib.layers.xavier_initializer(),
                                     activation=True,    
                                     name=name+'conv2')

        classifier = conv_layer(classifier, [1,1,32,128], model_state,
                                     use_bias=False,
                                     w_init=tf.contrib.layers.xavier_initializer(),
                                     activation=True,   
                                     name=name+'conv3')
        classifier = classifier + short
        classifier = conv_layer(classifier, [1,1,128,1], 
                                     b_shape=1,
                                     w_init=tf.truncated_normal_initializer(stddev=0.01),
                                     b_init=tf.constant_initializer(0),
                                     activation=False,  
                                     name=name+'conv4')      
        classifier = deconv.deconv(classifier, upscale)
        return classifier
def conv_layer(x, W_shape, model_state=True, b_shape=None, name=None, 
                   activation=None, bn = True,
                   padding='SAME', use_bias=True, w_init=None, b_init=None,
                   rate = None):
        with tf.variable_scope(name):
            W = weight_variable(W_shape, w_init)
            # tf.summary.histogram('weights_{}'.format(name), W)

            if use_bias:
                b = bias_variable([b_shape], b_init)
                # tf.summary.histogram('biases_{}'.format(name), b)
            if not rate:
                conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
            else:
                # print('shape=',W.get_shape().ndims)
                conv = tf.nn.atrous_conv2d(x, W, rate=rate, padding=padding)
                print('atrous done')
            if not activation:
                return conv + b if use_bias else conv
            else:
                if bn:
                    x = batch_norm(conv, train=model_state, 
                               eps=1e-05, decay=0.9, affine=True, name='BN')
                    return tf.nn.relu(x)
                else:
                    return tf.nn.relu(tf.nn.bias_add(conv,b))
# def get_conv_filter(name):
#     if 'conv' in name:
#         print('open conv5')
#         weights = tf.constant(ata_dict[name][0])
#         return tf.get_variable(initializer=weights, name="filter", regularizer=regularizer)
#     # else:
#     #     return tf.constant(ata_dict[name][0], name="filter")

# def get_bias(name):
#     if 'conv' in name:
#         biases = tf.constant(ata_dict[name][1])
#         return tf.get_variable(initializer=biases, name="biases")
#     # else:
#     #     return tf.constant(ata_dict[name][1], name="biases")

def weight_variable(shape, initial):

    init = initial(shape)
    # return tf.get_variable(initializer=init, regularizer=regularizer,name='weight')
    return tf.get_variable(initializer=init,name='weight')

def bias_variable(shape, initial):

    init = initial(shape)
    return tf.get_variable(initializer=init, name='biase')




def batch_norm(x, train, eps=1e-05, decay=0.9, affine=True, name=None):
    with tf.variable_scope(name, default_name='BatchNorm2d'):
        
        params_shape = x.get_shape().as_list()[-1]
        # depth = conv.get_shape().as_list()[-1]
        moving_mean = tf.get_variable('mean', params_shape,
                                      initializer=tf.constant_initializer(0.0),
                                      trainable=False)
        moving_variance = tf.get_variable('variance', params_shape,
                                          initializer=tf.constant_initializer(1.0),
                                          trainable=False)

        def mean_var_with_update():
            print('is training BN')
            mean, variance = tf.nn.moments(x, [0,1,2], name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        # print('here',params_shape)
        if train is not None:
          mean, variance = mean_var_with_update()
        else:
          print('is testing BN')
          mean, variance = moving_mean, moving_variance
        # mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))
        if affine :
            beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.constant_initializer(0.0))
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.constant_initializer(1.0))
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x