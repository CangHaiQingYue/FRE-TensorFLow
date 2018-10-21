import tensorflow as tf


def get_learning_rate(flags, global_step):

    decay_steps = flags.decay_steps
    if flags.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(flags.learning_rate,
                                          global_step,
                                          decay_steps,
                                          flags.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')

    elif flags.learning_rate_decay_type == 'fixed':
        return tf.constant(flags.learning_rate, name='fixed_learning_rate')
    elif flags.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(flags.learning_rate,
                                         global_step,
                                         decay_steps,
                                         flags.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         flags.learning_rate_decay_type)

def get_optimizer(flags, learning_rate):
    if flags.optimizer == 'adadelta':
        tf.logging.info('The optimizer is --> {}'.format(flags.optimizer)) 
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=flags.adadelta_rho,
            epsilon=flags.opt_epsilon)
    elif flags.optimizer == 'adagrad':
        tf.logging.info('The optimizer is --> {}'.format(flags.optimizer)) 
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=flags.adagrad_initial_accumulator_value)
    elif flags.optimizer == 'adam':
        tf.logging.info('The optimizer is --> {}'.format(flags.optimizer)) 
        optimizer = tf.train.AdamOptimizer(
            learning_rate)
    elif flags.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=flags.ftrl_learning_rate_power,
            initial_accumulator_value=flags.ftrl_initial_accumulator_value,
            l1_regularization_strength=flags.ftrl_l1,
            l2_regularization_strength=flags.ftrl_l2)
    elif flags.optimizer == 'momentum':
        tf.logging.info('The optimizer is --> {}'.format(flags.optimizer)) 
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=flags.momentum,
            name='Momentum')
    elif flags.optimizer == 'rmsprop':
        tf.logging.info('The optimizer is --> {}'.format(flags.optimizer)) 
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=flags.rmsprop_decay,
            momentum=flags.rmsprop_momentum,
            epsilon=flags.opt_epsilon)
    elif flags.optimizer == 'sgd':
        tf.logging.info('The optimizer is --> {}'.format(flags.optimizer)) 
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', flags.optimizer)
    return optimizer

def get_hooks(flags, summary_op, saver, loss):
    hooks = []
    checkpointhooks = tf.train.CheckpointSaverHook(checkpoint_dir=flags.checkpoint_dir,
                                            save_steps=flags.save_steps,
                                            saver=saver,
                                            checkpoint_basename=flags.checkpoint_basename)
    summaryhooks = tf.train.SummarySaverHook(save_steps=flags.save_steps,
                                            output_dir=flags.summary_dir,
                                            summary_op=summary_op)
    hooks.append(checkpointhooks)
    hooks.append(summaryhooks)
    hooks.append(tf.train.StopAtStepHook(last_step=flags.max_step))
    hooks.append(tf.train.NanTensorHook(loss))
    return hooks

def get_scaffold(flags, saver):
    """
    case 1: restore from FRE model 
          A. check the latest ckpt and assign
       
    case 2: restore from vgg16.npy
          A: replace name from FRE to vgg16
       in which:
                    flage.model_name = FRE
                    flage.ckpt_name = vgg16
    """

    if flags.fine_tuning:
        ckpt_path = flags.checkpoint_path
        print('ckpt_path_1=', ckpt_path)

        ckpt_path = tf.train.latest_checkpoint(ckpt_path)
        print('ckpt_path_2=', ckpt_path)
        if ckpt_path:
            variables_to_restore = tf.model_variables()
            init_assign_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(ckpt_path, \
                variables_to_restore, ignore_missing_vars=False)
            tf.logging.info('Restore checkpoint form {}'.format(ckpt_path))
            def InitAssignFn(scaffold,sess):
                sess.run(init_assign_op, init_feed_dict)
            return tf.train.Scaffold(init_fn=InitAssignFn)
        else:
            raise ValueError('There is no checkpoint to restore!!!')
    else:
        variables_to_restore = []
        variables_to_restore = {var.op.name.replace(flags.model_name,
                                flags.ckpt_name): var for var in tf.model_variables()}
       
        ckpt_path = flags.checkpoint_path
 
        vars_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(ckpt_path,
                variables_to_restore, ignore_missing_vars=True)
  
        init_assign_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
            ckpt_path, variables_to_restore, ignore_missing_vars=True)
        print('Restore Checkpoint From {}'.format(ckpt_path))
        def InitAssignFn(scaffold,sess):
            sess.run(init_assign_op, init_feed_dict)
            tf.logging.info('Network Init Done!')

        scaffold = tf.train.Scaffold(saver=saver, init_fn=InitAssignFn)
        return scaffold
       
