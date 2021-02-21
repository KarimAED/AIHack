import tensorflow as tf
import sys

from filter_norm import filter_norm_split
from model import CropModel


x_tr, x_te, y_tr, y_te = filter_norm_split()

crop_model = CropModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

loss_object = tf.keras.losses.MeanSquaredError()


def grad(model, inputs, targets):  # gradients
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)


# copy pasted from tensorflow
# Note: Rerunning this cell uses the same model variables

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = int(sys.argv[1])

for epoch in range(num_epochs):
  
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_mae = tf.keras.metrics.MeanAbsoluteError()
  
    # Training loop - using batches of 32
    for i in range(x_tr.shape[0]):
        
        sys.stdout.write(f'\r {i}   ')
        sys.stdout.flush()
        
        single_x = tf.expand_dims(x_tr[i], axis=0)
        single_y = tf.expand_dims(y_tr[i], axis=0)
        
        # Optimize the model
        loss_value, grads = grad(crop_model, single_x, single_y)
        optimizer.apply_gradients(zip(grads, crop_model.trainable_variables))
    
        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        epoch_mae.update_state(single_y, crop_model(single_x, training=True))
  
    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_mae.result())
  
    if epoch % 1 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, MAE: {:.3f}".format(epoch,
                                                               epoch_loss_avg.result(),
                                                               epoch_mae.result()))


loss_avg = tf.keras.metrics.Mean()
mae = tf.keras.metrics.MeanAbsoluteError()

for i in range(x_te.shape[0]):
    single_x = tf.expand_dims(x_te[i], axis=0)
    single_y = tf.expand_dims(y_te[i], axis=0)

    loss_value, grads = grad(crop_model, single_x, single_y)
    loss_avg.update_state(loss_value)  # Add current batch loss
    mae.update_state(single_y, crop_model(single_x))

print(f"Test performance:\n\tLoss: {loss_avg.result()}, MAE: {mae.result()}")
