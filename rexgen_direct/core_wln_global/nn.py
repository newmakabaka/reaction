import numpy as np
import tensorflow as tf
import math

tf.compat.v1.disable_v2_behavior()

def linear(input_, output_size, scope, reuse=False, init_bias=0.0):
    shape = input_.get_shape().as_list()
    stddev = min(1.0 / math.sqrt(shape[-1]), 0.1)
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        W = tf.compat.v1.get_variable("Matrix", [shape[-1], output_size], tf.compat.v1.float32, tf.compat.v1.random_normal_initializer(stddev=stddev))
    if init_bias is None:
        return tf.compat.v1.matmul(input_, W)
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        b = tf.compat.v1.get_variable("bias", [output_size], initializer=tf.compat.v1.constant_initializer(init_bias))
    return tf.compat.v1.matmul(input_, W) + b

def linearND(input_, output_size, scope, reuse=False, init_bias=0.0):
    shape = input_.get_shape().as_list()
    ndim = len(shape)
    stddev = min(1.0 / math.sqrt(shape[-1]), 0.1)
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        W = tf.compat.v1.get_variable("Matrix", [shape[-1], output_size], tf.compat.v1.float32, tf.compat.v1.random_normal_initializer(stddev=stddev))
    X_shape = tf.compat.v1.gather(tf.compat.v1.shape(input_), list(range(ndim-1)))
    target_shape = tf.compat.v1.concat([X_shape, [output_size]], 0)
    exp_input = tf.compat.v1.reshape(input_, [-1, shape[-1]])
    if init_bias is None:
        res = tf.compat.v1.matmul(exp_input, W)
    else:
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            b = tf.compat.v1.get_variable("bias", [output_size], initializer=tf.compat.v1.constant_initializer(init_bias))
        res = tf.compat.v1.matmul(exp_input, W) + b
    res = tf.compat.v1.reshape(res, target_shape)
    res.set_shape(shape[:-1] + [output_size])
    return res
