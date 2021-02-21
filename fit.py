import tensorflow as tf
import numpy as np
from filter_norm import filter_norm_split
from model import CropModel


x_tr, x_te, y_tr, y_te = filter_norm_split()

crop_model = CropModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


# loss function????


loss_object = tf.keras.losses.MeanSquaredError()#this should be the loss function


def grad(model, inputs, targets): # gradients
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)





xtt = tf.expand_dims(x_te[0],axis=0)
ytt= y_te[0]

def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)

  return loss_object(y_true=y, y_pred=y_)



loss_value, grads = grad(crop_model, xtt, ytt)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, crop_model.trainable_variables))

print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(crop_model, xtt, ytt, training=True).numpy()))


l = loss(crop_model, xtt, ytt, training=False)
print("Loss test: {}".format(l))
#%%


import sys




# copy pasted from tensorflow
## Note: Rerunning this cell uses the same model variables

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
  
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
  
    # Training loop - using batches of 32
    for i in range(x_tr.shape[0]):
        
        sys.stdout.write(f'\r {i}   ')
        sys.stdout.flush()
        
        single_x = tf.expand_dims(x_tr[i],axis=0)
        single_y = tf.expand_dims(y_tr[i],axis = 0)
        
        # Optimize the model
        loss_value, grads = grad(crop_model, single_x, single_y)
        optimizer.apply_gradients(zip(grads, crop_model.trainable_variables))
    
        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        epoch_accuracy.update_state(single_y, crop_model(single_x, training=True))
  
    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
  
    if epoch % 1 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
    

    
    
    
    
    
    
    