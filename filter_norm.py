import numpy as np

import tensorflow as tf


def inp_mean_std(x):
    x_n = tf.ragged.map_flat_values(tf.math.l2_normalize, x, axis=0)
    x_n = tf.ragged.map_flat_values(tf.math.l2_normalize, x_n, axis=0)
    x_n = tf.ragged.map_flat_values(tf.math.l2_normalize, x_n, axis=0)
    return x_n


def filter_norm_split():
    data = np.load("joint_set.npz", allow_pickle=True)

    inp = data["inp"]
    out = data["out"]

    out_mask = [1 if len(i) == 1 else 0 for i in out]
    inp_mask = [1 if len(i) != 0 else 0 for i in inp]

    mask1 = np.ma.make_mask(out_mask)
    mask2 = np.ma.make_mask(inp_mask)

    mask = mask1 & mask2

    inp = inp[mask]
    out = np.stack(out[mask]).flatten()

    inp = inp.tolist()

    inp = tf.ragged.constant(inp, inner_shape=(23, 2))

    # need to shuffle
    """shuffler = np.random.random(out.size)
    
    shuffler = np.argsort(shuffler)
    """

    x_train = inp[:1000, :, :, :]

    x_tr = inp_mean_std(x_train)

    x_test = inp[1000:, :, :, :]

    x_te = inp_mean_std(x_test)

    y_train = out[:1000]
    y_train -= np.mean(y_train)
    y_train /= np.std(y_train)
    y_test = out[1000:]
    y_test -= np.mean(y_test)
    y_test /= np.std(y_test)

    return x_tr, x_te, y_train, y_test
