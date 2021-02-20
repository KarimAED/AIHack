import numpy as np

import tensorflow as tf
from tensorflow.keras import layers


# event, location (within county), time (within year)
fake_data = np.random.random(size=(2000, 100, 20))
fake_target = np.random.random(size=2000)


print(fake_data)


class CropModel(tf.keras.Model):

    def __init__(self, filters=10, kernel_size=5):
        super(CropModel, self).__init__()
        self.conv_1 = layers.Conv1D(filters, kernel_size, padding="causal", input_shape=(100, 20))
        self.dense_1 = layers.Dense(20)
        self.dense_2 = layers.Dense(10)
        self.dense_out = layers.Dense(1)

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_out(x)
        return layers.AveragePooling1D(inputs.shape[1])(x)


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


model = CropModel()

model.compile(optimizer=optimizer, loss="mse", metrics="mae")

model.fit(fake_data, fake_target, epochs=100, batch_size=50, verbose=2)
