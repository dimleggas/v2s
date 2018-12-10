from __future__ import print_function
import cPickle as pickle
import os
import time

import numpy as np
import tensorflow as tf
from six.moves import xrange

import loader
from wavegan import WaveGANGenerator, WaveGANDiscriminator
from functools import reduce


"""
  Constants
"""
_FS = 16000
_WINDOW_LEN = 16384
_PRIOR_SIZE = 16*4096
_D_Z = 100

"""
  Trains a WaveGAN
"""
def train(aud, vid, args):
    print('setting up models...')
    with tf.name_scope('loader'):
        x = loader.get_batch(aud, args.train_batch_size, _WINDOW_LEN, args.data_first_window)
        y = loader.get_batch(vid, args.train_batch_size, _PRIOR_SIZE, args.data_first_window)

    # Make z vector
    z = tf.random_uniform([args.train_batch_size, _D_Z], -1., 1., dtype=tf.float32)

    # Make generator
    with tf.variable_scope('G'):
        G_z = WaveGANGenerator(z, y, train=True, **args.wavegan_g_kwargs)
        if args.wavegan_genr_pp:
            with tf.variable_scope('pp_filt'):
                G_z = tf.layers.conv1d(G_z, 1, args.wavegan_genr_pp_len, use_bias=False, padding='same')
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

    # Print G summary
    print('-' * 80)
    print('Generator vars')
    nparams = 0
    for v in G_vars:
        v_shape = v.get_shape().as_list()
        v_n = reduce(lambda x, y: x * y, v_shape)
        nparams += v_n
        print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
    print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))

    # Summarize
    tf.summary.audio('x', x, _FS)
    tf.summary.audio('G_z', G_z, _FS)
    G_z_rms = tf.sqrt(tf.reduce_mean(tf.square(G_z[:, :, 0]), axis=1))
    x_rms = tf.sqrt(tf.reduce_mean(tf.square(x[:, :, 0]), axis=1))
    tf.summary.histogram('x_rms_batch', x_rms)
    tf.summary.histogram('G_z_rms_batch', G_z_rms)
    tf.summary.scalar('x_rms', tf.reduce_mean(x_rms))
    tf.summary.scalar('G_z_rms', tf.reduce_mean(G_z_rms))

    # Make real discriminator
    with tf.name_scope('D_x'), tf.variable_scope('D'):
        D_x = WaveGANDiscriminator(x, y, **args.wavegan_d_kwargs)
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')

    # Print D summary
    print('-' * 80)
    print('Discriminator vars')
    nparams = 0
    for v in D_vars:
        v_shape = v.get_shape().as_list()
        v_n = reduce(lambda x, y: x * y, v_shape)
        nparams += v_n
        print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
    print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))
    print('-' * 80)

    # Make fake discriminator
    with tf.name_scope('D_G_z'), tf.variable_scope('D', reuse=True):
        D_G_z = WaveGANDiscriminator(G_z, y, **args.wavegan_d_kwargs)

    # Create loss
    D_clip_weights = None
    if args.wavegan_loss == 'dcgan':
        fake = tf.zeros([args.train_batch_size], dtype=tf.float32)
        real = tf.ones([args.train_batch_size], dtype=tf.float32)

        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          logits=D_G_z,
          labels=real
        ))

        D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          logits=D_G_z,
          labels=fake
        ))
        D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          logits=D_x,
          labels=real
        ))

        D_loss /= 2.
    elif args.wavegan_loss == 'lsgan':
        G_loss = tf.reduce_mean((D_G_z - 1.) ** 2)
        D_loss = tf.reduce_mean((D_x - 1.) ** 2)
        D_loss += tf.reduce_mean(D_G_z ** 2)
        D_loss /= 2.
    elif args.wavegan_loss == 'wgan':
        G_loss = -tf.reduce_mean(D_G_z)
        D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)

        with tf.name_scope('D_clip_weights'):
            clip_ops = []
            for var in D_vars:
                clip_bounds = [-.01, .01]
                clip_ops.append(
                  tf.assign(
                    var,
                    tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                  )
                )
            D_clip_weights = tf.group(*clip_ops)
    elif args.wavegan_loss == 'wgan-gp':
        G_loss = -tf.reduce_mean(D_G_z)
        D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)

        alpha = tf.random_uniform(shape=[args.train_batch_size, 1, 1], minval=0., maxval=1.)
        differences = G_z - x
        interpolates = x + (alpha * differences)
        with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
            D_interp = WaveGANDiscriminator(interpolates, y, **args.wavegan_d_kwargs)

        LAMBDA = 10
        gradients = tf.gradients(D_interp, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
        D_loss += LAMBDA * gradient_penalty
    else:
        raise NotImplementedError()

    tf.summary.scalar('G_loss', G_loss)
    tf.summary.scalar('D_loss', D_loss)

    # Create (recommended) optimizer
    if args.wavegan_loss == 'dcgan':
        G_opt = tf.train.AdamOptimizer(
            learning_rate=2e-4,
            beta1=0.5)
        D_opt = tf.train.AdamOptimizer(
            learning_rate=2e-4,
            beta1=0.5)
    elif args.wavegan_loss == 'lsgan':
        G_opt = tf.train.RMSPropOptimizer(
            learning_rate=1e-4)
        D_opt = tf.train.RMSPropOptimizer(
            learning_rate=1e-4)
    elif args.wavegan_loss == 'wgan':
        G_opt = tf.train.RMSPropOptimizer(
            learning_rate=5e-5)
        D_opt = tf.train.RMSPropOptimizer(
            learning_rate=5e-5)
    elif args.wavegan_loss == 'wgan-gp':
        G_opt = tf.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.5,
            beta2=0.9)
        D_opt = tf.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.5,
            beta2=0.9)
    else:
        raise NotImplementedError()

    # Create training ops
    G_train_op = G_opt.minimize(G_loss, var_list=G_vars,
        global_step=tf.train.get_or_create_global_step())
    D_train_op = D_opt.minimize(D_loss, var_list=D_vars)

    # Run training
    print('now training...')
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=args.train_dir,
        save_checkpoint_secs=args.train_save_secs,
        save_summaries_secs=args.train_summary_secs) as sess:
        epoch=0
        while epoch < 2000:
            epoch+=1
            start = time.time()
            # Train discriminator
            for i in xrange(args.wavegan_disc_nupdates):
                sess.run(D_train_op)
                # Enforce Lipschitz constraint for WGAN
                if D_clip_weights is not None:
                    sess.run(D_clip_weights)

            # Train generator
            sess.run(G_train_op)

            print('training epoch ' +  str(epoch) + ' took ' + str(time.time()-start) +'s')
            # test predictions
            if epoch % 100 == 0 and epoch > 0:
                np.savetxt('gen_drums/epoch'+str(epoch)+'.csv', G_z.eval(session=sess).reshape((64, 16384)), delimiter=',')
        sess.close()

"""
  Creates and saves a MetaGraphDef for simple inference
  Tensors:
    'samp_z_n' int32 []: Sample this many latent vectors
    'samp_z' float32 [samp_z_n, 100]: Resultant latent vectors
    'z:0' float32 [None, 100]: Input latent vectors
    'flat_pad:0' int32 []: Number of padding samples to use when flattening batch to a single audio file
    'G_z:0' float32 [None, 16384, 1]: Generated outputs
    'G_z_int16:0' int16 [None, 16384, 1]: Same as above but quantizied to 16-bit PCM samples
    'G_z_flat:0' float32 [None, 1]: Outputs flattened into single audio file
    'G_z_flat_int16:0' int16 [None, 1]: Same as above but quantized to 16-bit PCM samples
  Example usage:
    import tensorflow as tf
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph('infer.meta')
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession()
    saver.restore(sess, 'model.ckpt-10000')
    z_n = graph.get_tensor_by_name('samp_z_n:0')
    _z = sess.run(graph.get_tensor_by_name('samp_z:0'), {z_n: 10})
    z = graph.get_tensor_by_name('G_z:0')
    _G_z = sess.run(graph.get_tensor_by_name('G_z:0'), {z: _z})
"""
def infer(vid, args):
    print('inferring computational graph...')
    infer_dir = os.path.join(args.train_dir, 'infer')
    if not os.path.isdir(infer_dir):
        os.makedirs(infer_dir)

    # Subgraph that generates latent vectors
    samp_z_n = tf.placeholder(tf.int32, [], name='samp_z_n')
    samp_z = tf.random_uniform([samp_z_n, _D_Z], -1.0, 1.0, dtype=tf.float32, name='samp_z')

    # Input z0
    z = tf.placeholder(tf.float32, [None, _D_Z], name='z')
    flat_pad = tf.placeholder(tf.int32, [], name='flat_pad')

    # Input y0
    samp_y_n = tf.placeholder(tf.int32, [], name='samp_y_n')
    samp_y = loader.get_batch(vid, args.train_batch_size, _PRIOR_SIZE, args.data_first_window)
    samp_y = tf.identity(samp_y, name='samp_y')
    y = tf.placeholder(tf.float32, [None, _PRIOR_SIZE, 1], name='y')

    # Execute generator
    with tf.variable_scope('G'):
        G_z = WaveGANGenerator(z, y, train=False, **args.wavegan_g_kwargs)
        if args.wavegan_genr_pp:
            with tf.variable_scope('pp_filt'):
                G_z = tf.layers.conv1d(G_z, 1, args.wavegan_genr_pp_len, use_bias=False, padding='same')
    G_z = tf.identity(G_z, name='G_z')

    # Flatten batch
    nch = int(G_z.get_shape()[-1])
    G_z_padded = tf.pad(G_z, [[0, 0], [0, flat_pad], [0, 0]])
    G_z_flat = tf.reshape(G_z_padded, [-1, nch], name='G_z_flat')

    # Encode to int16
    def float_to_int16(x, name=None):
        x_int16 = x * 32767.
        x_int16 = tf.clip_by_value(x_int16, -32767., 32767.)
        x_int16 = tf.cast(x_int16, tf.int16, name=name)
        return x_int16
    G_z_int16 = float_to_int16(G_z, name='G_z_int16')
    G_z_flat_int16 = float_to_int16(G_z_flat, name='G_z_flat_int16')

    # Create saver
    G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G')
    global_step = tf.train.get_or_create_global_step()
    saver = tf.train.Saver(G_vars + [global_step])

    # Export graph
    tf.train.write_graph(tf.get_default_graph(), infer_dir, 'infer.pbtxt')

    # Export MetaGraph
    infer_metagraph_fp = os.path.join(infer_dir, 'infer.meta')
    tf.train.export_meta_graph(
        filename=infer_metagraph_fp,
        clear_devices=True,
        saver_def=saver.as_saver_def())

    # Reset graph (in case training afterwards)
    tf.reset_default_graph()


if __name__ == '__main__':
    import argparse
    import glob
    import sys

    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str, choices=['train', 'preview', 'incept', 'infer'])
    parser.add_argument('train_dir', type=str,
        help='Training directory')

    data_args = parser.add_argument_group('Data')
    data_args.add_argument('--data_dir', type=str,
        help='Data directory')
    data_args.add_argument('--data_first_window', action='store_true', dest='data_first_window',
        help='If set, only use the first window from each audio example')

    wavegan_args = parser.add_argument_group('WaveGAN')
    wavegan_args.add_argument('--wavegan_kernel_len', type=int,
        help='Length of 1D filter kernels')
    wavegan_args.add_argument('--wavegan_dim', type=int,
        help='Dimensionality multiplier for model of G and D')
    wavegan_args.add_argument('--wavegan_batchnorm', action='store_true', dest='wavegan_batchnorm',
        help='Enable batchnorm')
    wavegan_args.add_argument('--wavegan_disc_nupdates', type=int,
        help='Number of discriminator updates per generator update')
    wavegan_args.add_argument('--wavegan_loss', type=str, choices=['dcgan', 'lsgan', 'wgan', 'wgan-gp'],
        help='Which GAN loss to use')
    wavegan_args.add_argument('--wavegan_genr_upsample', type=str, choices=['zeros', 'nn', 'lin', 'cub'],
        help='Generator upsample strategy')
    wavegan_args.add_argument('--wavegan_genr_pp', action='store_true', dest='wavegan_genr_pp',
        help='If set, use post-processing filter')
    wavegan_args.add_argument('--wavegan_genr_pp_len', type=int,
        help='Length of post-processing filter for DCGAN')
    wavegan_args.add_argument('--wavegan_disc_phaseshuffle', type=int,
        help='Radius of phase shuffle operation')

    train_args = parser.add_argument_group('Train')
    train_args.add_argument('--train_batch_size', type=int,
        help='Batch size')
    train_args.add_argument('--train_save_secs', type=int,
        help='How often to save model')
    train_args.add_argument('--train_summary_secs', type=int,
        help='How often to report summaries')

    preview_args = parser.add_argument_group('Preview')
    preview_args.add_argument('--preview_n', type=int,
        help='Number of samples to preview')

    incept_args = parser.add_argument_group('Incept')
    incept_args.add_argument('--incept_metagraph_fp', type=str,
        help='Inference model for inception score')
    incept_args.add_argument('--incept_ckpt_fp', type=str,
        help='Checkpoint for inference model')
    incept_args.add_argument('--incept_n', type=int,
        help='Number of generated examples to test')
    incept_args.add_argument('--incept_k', type=int,
        help='Number of groups to test')

    parser.set_defaults(
      data_dir=None,
      data_first_window=False,
      wavegan_kernel_len=25,
      wavegan_dim=32,
      wavegan_batchnorm=False,
      wavegan_disc_nupdates=5,
      wavegan_loss='wgan-gp',
      wavegan_genr_upsample='zeros',
      wavegan_genr_pp=False,
      wavegan_genr_pp_len=512,
      wavegan_disc_phaseshuffle=2,
      train_batch_size=64,
      train_save_secs=600,
      train_summary_secs=240,
      preview_n=32,
      incept_metagraph_fp='./eval/inception/infer.meta',
      incept_ckpt_fp='./eval/inception/best_acc-103005',
      incept_n=5000,
      incept_k=10)

    args = parser.parse_args()

    # Make train dir
    if not os.path.isdir(args.train_dir):
        os.makedirs(args.train_dir)

    # Save args
    with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # Make model kwarg dicts
    setattr(args, 'wavegan_g_kwargs', {
        'kernel_len': args.wavegan_kernel_len,
        'dim': args.wavegan_dim,
        'use_batchnorm': args.wavegan_batchnorm,
        'upsample': args.wavegan_genr_upsample
    })
    setattr(args, 'wavegan_d_kwargs', {
        'kernel_len': args.wavegan_kernel_len,
        'dim': args.wavegan_dim,
        'use_batchnorm': args.wavegan_batchnorm,
        'phaseshuffle_rad': args.wavegan_disc_phaseshuffle
    })

    # Assign appropriate split for mode
    if args.mode == 'train':
        split = ('audio', 'video')
    else:
        split = None

    if split is not None:
        aud = glob.glob(os.path.join(args.data_dir, split[0]) + '*.tfrecord')
        vid = glob.glob(os.path.join(args.data_dir, split[1]) + '*.tfrecord')
        aud.sort()
        vid.sort()
        print(aud[0])
        print(vid[0])

        aud_test = glob.glob(os.path.join(args.data_dir, 'test_', split[0]) + '*.tfrecord')
        vid_test = glob.glob(os.path.join(args.data_dir, 'test_', split[1]) + '*.tfrecord')
        aud_test.sort()
        vid_test.sort()

    if args.mode == 'train':
        infer(vid, args)
        tf.reset_default_graph()
        train(aud, vid, args)
    elif args.mode == 'infer':
        infer(args)
    else:
        raise NotImplementedError()
