import numpy as np

import tensorflow as tf
from tensorflow.keras import layers


# event, location (within county), time (within year)
fake_data = np.random.random(size=(2000, 20, 20, 2))
fake_target = np.random.random(size=2000)


class CropModel(tf.keras.Model):

    def __init__(self, filters=1, kernel_size=5):
        super(CropModel, self).__init__()
        self.inp = layers.InputLayer(input_shape=[None, 23, 2], ragged=True)
        self.conv_1 = layers.Conv1D(filters, kernel_size, input_shape=(20, 2))
        self.dense_1 = layers.Dense(20)
        self.dense_2 = layers.Dense(10)
        self.dense_out = layers.Dense(1)

    def call(self, inputs):
        x = self.inp(inputs)
        x = self.conv_1(x)
        x = x[:, :, :, 0]
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_out(x)
        return layers.AveragePooling1D(inputs.shape[1])(x)
