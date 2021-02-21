import tensorflow as tf
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np

from filter_norm import filter_norm_split

x_tr, x_te, y_tr, y_te = filter_norm_split()

x_tr_me = tf.reduce_mean(x_tr, axis=1)
x_te_me = tf.reduce_mean(x_te, axis=1)

x_tr_std = tf.reduce_mean(x_tr, axis=1)
x_te_std = tf.reduce_mean(x_te, axis=1)

x_tr = np.append(np.array(x_tr_me), np.array(x_tr_std), axis=2)
x_te = np.append(np.array(x_te_me), np.array(x_te_std), axis=2)

print(x_tr.shape)
print(x_te.shape)


new_model = tf.keras.models.Sequential()

new_model.add(layers.Conv1D(10, 10, activity_regularizer="l2"))
new_model.add(layers.Flatten())
new_model.add(layers.Dropout(rate=0.5))
new_model.add(layers.Dense(20, activation="relu", activity_regularizer="l2"))
new_model.add(layers.Dense(1))

opt = tf.keras.optimizers.Adadelta(learning_rate=1e-1)

new_model.compile(optimizer=opt, loss="mse", metrics=["mae"])

hist = new_model.fit(x_tr, y_tr, batch_size=32, validation_split=0.1, epochs=1000, verbose=2)

pt = new_model.evaluate(x_te, y_te)

plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.show()

plt.plot(hist.history["val_mae"])
plt.plot(hist.history["mae"])

plt.show()
