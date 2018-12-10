import numpy as np
import tensorflow as tf

def conv1d_transpose(inputs, filters, kernel_width, stride=4, padding='same', upsample='zeros'):
    if upsample == 'zeros':
        return tf.layers.conv2d_transpose(tf.expand_dims(inputs, axis=1),
                                          filters, (1, kernel_width),
                                          strides=(1, stride),
                                          padding='same')[:, 0]
    elif upsample == 'nn':
        batch_size = tf.shape(inputs)[0]
        _, w, nch = inputs.get_shape().as_list()

        x = inputs

        x = tf.expand_dims(x, axis=1)
        x = tf.image.resize_nearest_neighbor(x, [1, w * stride])
        x = x[:, 0]

        return tf.layers.conv1d(x, filters, kernel_width, 1, padding='same')
    else:
        raise NotImplementedError

"""
  Input: [None, 100]
  Output: [None, 159744, 1]
"""
def WaveGANGenerator(z, y, kernel_len=25, dim=64, use_batchnorm=False, upsample='zeros', train=False):
    batch_size = tf.shape(z)[0]

    if use_batchnorm:
        batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
    else:
        batchnorm = lambda x: x

    # FC and reshape for convolution
    # [100] -> [16, 1024]
    x = z
    with tf.variable_scope('z_project'):
        x = tf.layers.dense(x, 16 * dim * 16)
        x = tf.reshape(x, [batch_size, 16, dim * 16])
        x = batchnorm(x)
        x = tf.nn.relu(x)

    y = tf.reshape(y, [batch_size, 16*4096])
    with tf.variable_scope('y_project'):
        y = tf.layers.dense(y, 16 * dim * 16)
        y = tf.reshape(y, [batch_size, 16, dim * 16])
        y = batchnorm(y)
        y = tf.nn.relu(y)

    with tf.variable_scope('combine_z_y'):
        x = tf.math.add(x, y)
        x = batchnorm(x)
        x = tf.nn.relu(x)

    # Layer 0
    with tf.variable_scope('upconv_0'):
        x = conv1d_transpose(x, dim * 8, kernel_len, 4, upsample=upsample)
        x = batchnorm(x)
        x = tf.nn.relu(x)

    # Layer 1
    with tf.variable_scope('upconv_1'):
        x = conv1d_transpose(x, dim * 4, kernel_len, 4, upsample=upsample)
        x = batchnorm(x)
        x = tf.nn.relu(x)

    # Layer 2
    with tf.variable_scope('upconv_2'):
        x = conv1d_transpose(x, dim * 2, kernel_len, 4, upsample=upsample)
        x = batchnorm(x)
        x = tf.nn.relu(x)

    # Layer 3

    with tf.variable_scope('upconv_3'):
        x = conv1d_transpose(x, dim, kernel_len, 4, upsample=upsample)
        x = batchnorm(x)
        x = tf.nn.relu(x)

    # Layer 4
    with tf.variable_scope('upconv_4'):
        x = conv1d_transpose(x, 1, kernel_len, 4, upsample=upsample)
        x = tf.nn.tanh(x)

    # Automatically update batchnorm moving averages every time G is used during training
    if train and use_batchnorm:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if len(update_ops) != 10:
            raise Exception('Other update ops found in graph')
        with tf.control_dependencies(update_ops):
            x = tf.identity(x)

    return x


def lrelu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def apply_phaseshuffle(x, rad, pad_type='reflect'):
    b, x_len, nch = x.get_shape().as_list()

    phase = tf.random_uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
    pad_l = tf.maximum(phase, 0)
    pad_r = tf.maximum(-phase, 0)
    phase_start = pad_r
    x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

    x = x[:, phase_start:phase_start+x_len]
    x.set_shape([b, x_len, nch])

    return x

"""
  Input: [None, 16384, 1]
  Output: [None] (linear output)
"""

def WaveGANDiscriminator(x, y, kernel_len=25, dim=64, use_batchnorm=False, phaseshuffle_rad=0):
    batch_size = tf.shape(x)[0]

    if use_batchnorm:
        batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
    else:
        batchnorm = lambda x: x

    if phaseshuffle_rad > 0:
        phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
    else:
        phaseshuffle = lambda x: x

    # Layer 0
    # [16384, 1] -> [4096, 64]
    with tf.variable_scope('downconv_0'):
        x = tf.layers.conv1d(x, dim, kernel_len, 4, padding='SAME')
    x = lrelu(x)
    x = phaseshuffle(x)

    # Layer 1
    # [4096, 64] -> [1024, 128]
    with tf.variable_scope('downconv_1'):
        x = tf.layers.conv1d(x, dim * 2, kernel_len, 4, padding='SAME')
        x = batchnorm(x)
    x = lrelu(x)
    x = phaseshuffle(x)

    # Layer 2
    # [1024, 128] -> [256, 256]
    with tf.variable_scope('downconv_2'):
        x = tf.layers.conv1d(x, dim * 4, kernel_len, 4, padding='SAME')
        x = batchnorm(x)
    x = lrelu(x)
    x = phaseshuffle(x)

    # Layer 3
    # [256, 256] -> [64, 512]
    with tf.variable_scope('downconv_3'):
        x = tf.layers.conv1d(x, dim * 8, kernel_len, 4, padding='SAME')
        x = batchnorm(x)
    x = lrelu(x)
    x = phaseshuffle(x)

    # Layer 4
    # [64, 512] -> [16, 1024]
    with tf.variable_scope('downconv_4'):
        x = tf.layers.conv1d(x, dim * 16, kernel_len, 4, padding='SAME')
        x = batchnorm(x)
    x = lrelu(x)

    # Flatten
    x = tf.reshape(x, [batch_size, 16 * dim * 16])
    x = tf.layers.dense(x, 512)
    x = batchnorm(x)
    x = tf.nn.relu(x)

    y = tf.reshape(y, [batch_size, 16*4096])
    y = tf.layers.dense(y, 512)
    y = batchnorm(y)
    y = tf.nn.relu(y)

    x = tf.concat([x, y], -1)
    x = tf.layers.dense(x, 512)
    x = batchnorm(x)
    x = tf.nn.relu(x)

    # Connect to single logit
    with tf.variable_scope('output'):
        x = tf.layers.dense(x, 1)[:, 0]

    # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training
    return x
