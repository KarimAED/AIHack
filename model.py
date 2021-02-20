import numpy as np

import tensorflow as tf
from tensorflow.keras import layers


fake_data = np.random.random(size=(2000, 100, 20))
fake_target = np.random.random(size=2000)


print(fake_data)


class CropModel(tf.keras.Model):

    def __init__(self):
        super(self).__init__()
        self.conv_1 = layers.Conv1D()
        self.dense_1 = layers.Dense()
        self.dense_2 = layers.Dense()
        self.dense_out = layers.Dense(1)

    def call(self, inputs):
        print(inputs.shape)
        x = self.conv_1(inputs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return self.dense_out(x)


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
