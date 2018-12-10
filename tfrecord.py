import sys
import numpy as np
import tensorflow as tf

def np_to_tfrecords(X, Y, file_path_prefix, verbose=True):
    def _dtype_feature(ndarray):
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        else:
            raise ValueError("The input should be numpy ndarray. \
                               Instaed got {}".format(ndarray.dtype))

    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 2  # If X has a higher rank,
                               # it should be rshape before fed to this function.
    assert isinstance(Y, np.ndarray) or Y is None

    # load appropriate tf.train.Feature class depending on dtype
    dtype_feature_x = _dtype_feature(X)
    if Y is not None:
        assert X.shape[0] == Y.shape[0]
        assert len(Y.shape) == 2
        dtype_feature_y = _dtype_feature(Y)

    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(result_tf_file)
    if verbose:
        print "Serializing {:d} examples into {}".format(X.shape[0], result_tf_file)

    # iterate over each sample,
    # and serialize it as ProtoBuf.
    for idx in range(X.shape[0]):
        x = X[idx]
        if Y is not None:
            y = Y[idx]

        d_feature = {}
        d_feature['X'] = dtype_feature_x(x)
        if Y is not None:
            d_feature['Y'] = dtype_feature_y(y)

        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)

    if verbose:
        print "Writing {} done!".format(result_tf_file)

dataset = sys.argv[1]
num_to_read = int(sys.argv[2])

for i in range(num_to_read / 64):
    batch = np.zeros((64, 156*4096))
    for j in range(64):
        vid = np.load('data/'+dataset+'/vid_feats/video_%05d.npz'%(64*i+j))['features']
        batch[j] = vid.ravel()
    np_to_tfrecords(batch, None, './data/'+dataset+'/tfr/video_%02d'%i, verbose=True)

    batch = np.zeros((64, 159744))
    for j in range(64):
        aud = np.load('data/'+dataset+'/aud_feats/video_%05d.npz'%(64*i+j))['features']
        batch[j] = aud.mean(0)
    np_to_tfrecords(batch, None, './data/'+dataset+'/tfr/audio_%02d'%i, verbose=True)
