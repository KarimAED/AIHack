import tensorflow as tf
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np

from filter_norm import filter_norm_split

x_tr, x_te, y_tr, y_te = filter_norm_split()

new_model = tf.keras.models.Sequential()

new_model.add(layers.Conv1D(3, 10, activity_regularizer="l2"))
new_model.add(layers.Dropout(rate=0.25))
new_model.add(layers.Flatten())
new_model.add(layers.Dropout(rate=0.25))
new_model.add(layers.Dense(16, activation="relu", activity_regularizer="l2"))
new_model.add(layers.Dropout(rate=0.25))
new_model.add(layers.Dense(16, activation="relu", activity_regularizer="l2"))
new_model.add(layers.Dense(1, activation="linear"))

opt = tf.keras.optimizers.Adamax(learning_rate=0.1)
new_model.compile(optimizer=opt, loss="mse", metrics=["mae"])

stopper = tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=20, min_delta=0.0001)


hist = new_model.fit(x_tr, y_tr, batch_size=10, shuffle=True, validation_split=0.1,
                     epochs=1000, verbose=2, callbacks=[stopper])

pt = new_model.evaluate(x_te, y_te)

print(f"Testing: Loss: {pt[0]}, MAE:{pt[1]}")

x = np.append(x_tr, x_te, axis=0)
y = np.append(y_tr, y_te, axis=0)

pred_tr = new_model.predict(x_tr)
pred_te = new_model.predict(x_te)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
ax1, ax2, ax3 = ax
ax1.plot(hist.history["loss"], label="Training Set")
ax1.plot(hist.history["val_loss"], label="Validation Set")
ax1.grid()
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(hist.history["val_mae"], label="Training Set")
ax2.plot(hist.history["mae"], label="Validation Set")
ax2.grid()
ax2.set_xlabel("Epochs")
ax2.set_ylabel("MAE")
ax2.legend()

ax3.scatter(y_tr, pred_tr.flatten(), s=1, alpha=0.3)
ax3.scatter(y_te, pred_te.flatten(), s=1, alpha=0.3)
ax3.grid()
x = np.arange(np.min(y), np.max(y), 0.01)
ax3.set_xlabel("Labels")
ax3.set_ylabel("Predictions")
ax3.plot(x, x, "k--")

plt.show()
