import tensorflow as tf
from model import deconv
from model.loss import fre_loss

# add_arg_scope=tf.contrib.framework.add_arg_scope
slim = tf.contrib.slim


class FRE():
    def __init__(self, run_state):
        tf.logging.info('The model is from fre_modle.py')
        if run_state == 'training':
            self.model_state = True
        else:
            self.model_state = False
    def net(self, model_name, inputs):
        return fre_net(model_name, inputs,self.model_state)
    def losses(self, end_points, labels):
        return fre_losses(end_points, labels)



# regularizer = tf.contrib.layers.l2_regularizer(0.0002)
regularizer = None

def fre_net(model_name, inputs, model_state):
    with tf.variable_scope(model_name,'FRE', [inputs]):
        end_points = {}
        with slim.arg_scope([slim.conv2d],weights_regularizer=regularizer,
                                            biases_regularizer=None):
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
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], 
                                    strides=[1, 1, 1, 1], padding='SAME', name='pool4')
            # net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.conv2d(net, num_outputs=512, kernel_size=3, stride=1, rate=2, scope='conv5/conv5_1')
            net = slim.conv2d(net, num_outputs=512, kernel_size=3, stride=1, rate=2, scope='conv5/conv5_2')
            net = slim.conv2d(net, num_outputs=512, kernel_size=3, stride=1, rate=2, scope='conv5/conv5_3')
            end_points['stage5'] = net
            tf.logging.info('Backbone Configure Done!')
        #  get side_output layer.  For block_1, just performing mapping without deconvolution
        with tf.variable_scope('freBlock_1'):
            freBlock_1 = get_fre_block(end_points['stage1'],
                                   filters=1, kernel_size=1, activation=None)

        freBlock_2 = side_layer(end_points['stage2'], upscale=2,
                                name='freBlock_2', model_state=model_state)
        freBlock_3 = side_layer(end_points['stage3'], upscale=4,
                                name='freBlock_3', model_state=model_state)
        freBlock_4 = side_layer(end_points['stage4'], upscale=8,
                                name='freBlock_4', model_state=model_state)
        freBlock_5 = side_layer(end_points['stage5'], upscale=8,
                                name='freBlock_5', model_state=model_state)
        fuse = [freBlock_1, freBlock_2, freBlock_3, freBlock_4, freBlock_5]
        fuse = get_fre_block(tf.concat(fuse, 3), filters=1, 
                                kernel_size=1, activation=None)

        side_output = [freBlock_1, freBlock_2, freBlock_3, freBlock_4,
                                freBlock_5, fuse]
        tf.logging.info('Side_output Configure Done!')
        tf.summary.image('fuse_image', tf.nn.sigmoid(fuse))

        return side_output

def fre_losses(end_points, labels):
    loss = 0.0
    if len(end_points) < 6:
        raise ValueError('Master: lacking of a layers!')
    with tf.variable_scope('loss'):
        for i, key in enumerate(end_points):         
            cost = fre_loss(key, labels)
            loss += cost
    tf.losses.add_loss(loss)
    
    if loss is not None:
        tf.logging.info('Add_loss Done!')
        tf.summary.scalar('weights_loss', loss)



def get_fre_block(net, filters, kernel_size, use_bias=True, activation=None, model_state=True):
    '''
      There are two case:
            case1: act_fu=True, use_bias=False.     this satuation was applied at: 1*1*128 1*1*32 3*3*32
                                                                                    and side_output_1
            case2: act_fu=False, use_bias=True.     this kind of satuation as applied at:
                                                            fuse_layer and the layer after deconv
    '''

    if not activation:     
        output = tf.layers.conv2d(
                    inputs=net,
                    filters=filters,
                    kernel_size=kernel_size,
                    padding='same',
                    activation=None,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    kernel_regularizer=regularizer,
                    use_bias=True,
                    bias_regularizer=None,
                    bias_initializer=tf.zeros_initializer(),
                    activity_regularizer=None)
        return output
    else:
        output = tf.layers.conv2d(inputs=net,
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    padding='same',
                                    activation=None,
                                    use_bias=False,
                                    bias_initializer=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    kernel_regularizer=regularizer)
        output = tf.layers.batch_normalization(output, momentum=0.9,
                    epsilon=1e-5,
                    training=model_state,
                    beta_regularizer=None,
                    gamma_regularizer=None
                    )

        output = tf.nn.relu(output)
        return output

def side_layer(inputs, upscale, name, model_state):
    
    with tf.variable_scope(name):
        shortcut = get_fre_block(inputs, filters=128, kernel_size=1,
                                    activation=True,model_state=model_state)
        
        net = get_fre_block(inputs, filters=32, kernel_size=1,
                                    activation=True,model_state=model_state)
        net = get_fre_block(net, filters=32, kernel_size=3,
                                    activation=True,model_state=model_state)
        net = get_fre_block(net, filters=128, kernel_size=1,
                                    activation=True,model_state=model_state)
        
        net = net + shortcut
        net = get_fre_block(net, filters=1, kernel_size=1,
                                    activation=None,model_state=model_state)
        net = deconv.deconv(net, upscale)
        

        return net
