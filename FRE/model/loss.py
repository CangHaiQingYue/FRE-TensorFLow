import tensorflow as tf


def fre_loss(logits, label, name='cross_entropy_loss'):
    y = tf.cast(label, tf.float32)
    r = 1
    n = 1.0
    a = 1.0
    count_neg = tf.reduce_sum((1.0 - y), axis=[1,2,3], keepdims=True) + 1e-5
    count_pos = tf.reduce_sum(y, axis=[1,2,3], keepdims=True) + 1e-5
    beta = tf.divide(count_neg, tf.add(count_neg, count_pos))
    # pos_weight = beta / (1 - beta)
    hed_pos_weight = tf.convert_to_tensor(count_neg / count_pos)   # = beta / (1-beta)

    h = tf.nn.sigmoid(logits)
    pos_weight = tf.multiply(hed_pos_weight, tf.pow(((a)/(h+n)),r))  # pos_weight * tf.pow(((1.0)/(h+n)),r)

    cost = tf.nn.weighted_cross_entropy_with_logits(targets=y,logits=logits, pos_weight=pos_weight)
    
    # condition1 = label <= 0.5
    # condition2 = label > 0.0
    # condition = tf.logical_and(condition1, condition2)

    
    # c = tf.where(condition, tf.zeros(shape=tf.shape(label)), tf.ones(shape=tf.shape(label)))
    # cost = c * cost

    cost = tf.reduce_mean(tf.multiply(tf.multiply(cost,tf.pow((h+n),r)),(1 - beta)))  # cost * (1-beta) * tf.pow((n+h),r)
    # print('loss_function = new focal loss')

    return cost
###########################
    # r = 2
    # n = 0.5
    # a = 1.0
    # y = tf.cast(label, tf.float32)
    # y_0 = tf.zeros_like(y)

    # beta = tf.reduce_mean(tf.cast(tf.equal(y, y_0), tf.float32), axis=[1,2,3], keepdims=True)
    # beta = beta + 1e-5
    # h = tf.nn.sigmoid(logits) + 1e-5
    # pos_weight = ((1 - beta) / beta) * tf.pow(((1 - h) / h), r)

    
    # cost = tf.nn.weighted_cross_entropy_with_logits(targets=y,logits=logits, pos_weight=pos_weight)
    # cost = cost * beta * tf.pow(h, 2)

    # condition1 = label <= 0.5
    # condition2 = label > 0.0
    # condition = tf.logical_and(condition1, condition2)
    # c = tf.where(condition, tf.zeros(shape=tf.shape(label)), tf.ones(shape=tf.shape(label)))
    
    # cost = tf.multiply(c, cost)
    # cost = tf.reduce_mean(cost)  # cost * (1-beta) * tf.pow((n+h),r)
    # # # print('loss_function = new focal loss')

    # return cost
    # ################### RCF Loss
    # y = tf.cast(label, tf.float32)
    # y_0 = tf.zeros_like(y)

    # beta = tf.reduce_mean(tf.cast(tf.equal(y, y_0), tf.float32))
    # alpha = 1.1 * (1 - beta)
    
    # pos_weight = (1.0 / 1.1) * (beta / (1 - beta))
    # cost = tf.nn.weighted_cross_entropy_with_logits(targets=y,logits=logits, pos_weight=pos_weight)
    # cost = alpha * cost
    # # when 0<y<=0.5 loss = 0.0
    # # condition1 = y <= 0.5
    # # condition2 = y > 0.0
    # # condition = tf.logical_and(condition1, condition2)
    # # c = tf.where(condition, tf.zeros(shape=tf.shape(y)), tf.ones(shape=tf.shape(y)))

    # # cost = c * cost
    # cost = tf.reduce_mean(cost)

    # return cost
