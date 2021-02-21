import numpy as np

import tensorflow as tf
from tensorflow.keras import layers


class CropModel(tf.keras.Model):

    def __init__(self, filters=20, kernel_size=5):
        super(CropModel, self).__init__()
        self.conv_1 = layers.Conv1D(filters, kernel_size, input_shape=(23, 2))
        self.dense_1 = layers.Dense(20, activation="relu")
        self.dense_2 = layers.Dense(10, activation="relu")
        self.dense_out = layers.Dense(1)

    def call(self, inputs):
        x = tf.reduce_mean(inputs, axis=1)
        x = self.conv_1(x)
        x = layers.GlobalMaxPool1D(data_format="channels_last")(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return self.dense_out(x)
