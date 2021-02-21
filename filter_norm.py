import numpy as np
import random
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

    random.seed(0)
    random.shuffle(inp)
    random.seed(0)
    random.shuffle(out)

    inp = tf.ragged.constant(inp, inner_shape=(23, 2))

    x_train = inp[:1000, :, :, :]

    x_tr = inp_mean_std(x_train)

    x_test = inp[1000:, :, :, :]

    x_te = inp_mean_std(x_test)

    x_tr_me = tf.reduce_mean(x_tr, axis=1)
    x_te_me = tf.reduce_mean(x_te, axis=1)

    x_tr_std = tf.reduce_mean(x_tr, axis=1)
    x_te_std = tf.reduce_mean(x_te, axis=1)

    x_tr = np.append(np.array(x_tr_me), np.array(x_tr_std), axis=2)
    x_te = np.append(np.array(x_te_me), np.array(x_te_std), axis=2)

    y_train = out[:1000]
    y_tr_m = np.mean(y_train)
    y_tr_std = np.std(y_train)
    y_train -= y_tr_m
    y_train /= y_tr_std
    y_test = out[1000:]
    y_te_m = np.mean(y_test)
    y_te_std = np.std(y_test)
    y_test -= y_te_m
    y_test /= y_te_std

    print(y_tr_m, y_tr_std, y_te_m, y_te_std)

    return x_tr, x_te, y_train, y_test
