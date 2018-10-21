import os
import time
import tensorflow as tf
import tf_utils
from data_provider import data_provide
from model import fre_model
from pprint import pprint
# from model import try_model_1 as fre_model
# from model import new_model as fre_model
# ====================dataset======================#
tf.app.flags.DEFINE_string('tfrecord_path', '/home/liupengli/myWork/FRE/FRE/tfrecord',
                            'floder name, where tfrecord in')
tf.app.flags.DEFINE_integer('mini_batch', 10,
                            'mini_btach size when training')
#====================learning_rate====================#
tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential',
                            'learning_rate  name')
tf.app.flags.DEFINE_float('learning_rate', 0.001,
                            'base learning_rate')
tf.app.flags.DEFINE_float('decay_steps', 5000,
                            'steps which learning_rate will be decayed')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.32,
                            'every 5000 steps lr will be multipied by ')
tf.app.flags.DEFINE_float('end_learning_rate', 1e-8,
                            'end of learning_rate')
#====================optimizer====================#
#==================================================#
tf.app.flags.DEFINE_string('optimizer', 'adam',
                            'optimizer  name')
tf.app.flags.DEFINE_float('adadelta_rho', 0.95,
                            'The decay rate for adadelta.')
tf.app.flags.DEFINE_float('adagrad_initial_accumulator_value', 0.1,
                            'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float( 'adam_beta1', 0.9,
                            'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float('adam_beta2', 0.999,
                            'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1e-8,
                            'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                            'The learning rate power.')
tf.app.flags.DEFINE_float('ftrl_initial_accumulator_value', 0.1,
                            'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float('ftrl_l1', 0.0,
                           'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float('ftrl_l2', 0.0,
                           'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float('momentum', 0.9,
                            'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')
#====================initialize====================#
#==================================================#

tf.app.flags.DEFINE_bool('fine_tuning', False, 'whether restore from ckpt or vgg16')
tf.app.flags.DEFINE_string('checkpoint_path','/home/liupengli/myWork/FRE/checkpoints',
                            'folder name where the checkpoint save in')
tf.app.flags.DEFINE_string('checkpoint_dir','/home/liupengli/myWork/FRE/checkpoints',
                            'folder name, checkpoint save in During training')
tf.app.flags.DEFINE_float('save_steps', 100,
                            'every N steps the ckpt and summary will save')
tf.app.flags.DEFINE_string('checkpoint_basename', 'fre.ckpt',
                            'ckpt basename for saving')
tf.app.flags.DEFINE_string('summary_dir', '/home/liupengli/myWork/FRE/summaries',
                            'folder name, summary save in During training')
tf.app.flags.DEFINE_float('max_step', 42000,
                            'the model will train for N steps')

#====================During training====================#
#=======================================================#
tf.app.flags.DEFINE_string('run_state', 'training',
                            'model state: training or testing')
tf.app.flags.DEFINE_string('model_name', 'FRE','model name')
tf.app.flags.DEFINE_string('ckpt_name', 'vgg_16','name of vgg16.npy')
#=======================================================#

flags = tf.app.flags.FLAGS

def main(_):
    # tf.reset_default_graph()

    with tf.Graph().as_default():

        tf.logging.set_verbosity(tf.logging.INFO)
        ## first of all, there should have a global_step
        global_step = tf.train.create_global_step()

        ## get dataset from tf.dataset
        images, labels = data_provide.get_data(flags)

        # print('imageshape=', images.get_shape())
        # print('labelshape=', labels.get_shape())
        ## difine model and get losses
        net_init = fre_model.FRE(flags.run_state)
        end_points = net_init.net(flags.model_name, images)
        net_init.losses(end_points, labels)
        total_loss = tf.losses.get_total_loss(
            add_regularization_losses=True, name='total_loss')
        regular_loss = tf.losses.get_regularization_loss()

        learning_rate = tf_utils.get_learning_rate(flags, global_step)
        optimizer = tf_utils.get_optimizer(flags, learning_rate)

        tf.summary.scalar('regular_loss', regular_loss)
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.image('inputs', images)
        tf.summary.image('groundTruth', labels)

        # generate hooks
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()
        hooks = tf_utils.get_hooks(flags, summary_op, saver, total_loss)

        # generate scaffold
        scaffold = tf_utils.get_scaffold(flags, saver)
        #########3
       
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_op):
            train_op = optimizer.minimize(total_loss, global_step=global_step)
        if train_op is None:
            raise ValueError
        # pprint(flags.__flags, stream=None)
        with tf.train.MonitoredTrainingSession(is_chief=True, 
                hooks=hooks,
                scaffold=scaffold
                 ) as sess:

            i = 0            
            print('Master--->Traing Started at ', time.strftime('%c'))
            while not sess.should_stop():
                try:
                    _, training_loss= sess.run([train_op, 
                                                total_loss
                                                ])
                    if i % 10 == 0:
                        tf.logging.info('Step = {step} loss = {loss}'.format(
                                                loss=training_loss,
                                                step=i))
                    i += 1
                    if i == flags.max_step:
                        print('Master--->Traing Done at ', time.strftime('%c'))
                except RuntimeError:
                    print("deal Run called even after should_stop requested")
                    break

        



if __name__ == '__main__':
    tf.app.run()
    print('========================')
