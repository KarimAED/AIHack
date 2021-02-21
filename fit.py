import tensorflow as tf
import numpy as np

from filter_norm import filter_norm_split
from model import CropModel

x_tr, x_te, y_tr, y_te = filter_norm_split()

crop_model = CropModel()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

