# DO NOT CHANGE THIS CELL

# We first import TensorFlow and other libraries
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# We then set up some functions and local variables
predictions = []
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    predictions.append(model.predict(xs))
callbacks = myCallback()

# We then define the xs (inputs) and ys (outputs)
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)


# PLEASE COMPELTE THIS CELL

SHAPE = [1] #YOUR CODE HERE#
LOSS = 'mean_squared_error' #YOUR CODE HERE#

# Define your model type
model = Sequential([Dense(units=1, input_shape=SHAPE)])
    
# Compile your model with choice of optimizer and loss function
model.compile(optimizer='sgd', loss=LOSS)

# DO NOT CHANGE THIS CELL

# We then fit the model
model.fit(xs, ys, epochs=300, callbacks=[callbacks], verbose=2)

EPOCH_NUMBERS=[1,25,50,150,300] # Update me to see other Epochs
plt.plot(xs,ys,label = "Ys")
for EPOCH in EPOCH_NUMBERS:
    plt.plot(xs,predictions[EPOCH-1],label = "Epoch = " + str(EPOCH))
plt.legend()
plt.show()




