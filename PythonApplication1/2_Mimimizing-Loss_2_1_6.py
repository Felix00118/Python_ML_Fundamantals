# First import the functions we will need
from __future__ import absolute_import, division, print_function, unicode_literals
from ctypes.wintypes import DWORD

#try:
#  # %tensorflow_version only exists in Colab.
#    %tensorflow_version 2.x
#except Exception:
#    pass
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Define our initial guess
INITIAL_W = 10.0
INITIAL_B = 10.0

# Define our loss function
def loss(predicted_y, target_y):
  return tf.reduce_mean(tf.square(predicted_y - target_y))

# Define our training procedure
def train(model, inputs, outputs, learning_rate):
  with tf.GradientTape() as t:
    current_loss = loss(model(inputs), outputs)
    # Here is where you differentiate the model values with respect to the loss function
    dw, db = t.gradient(current_loss, [model.w, model.b])
    # And here is where you update the model values based on the learning rate chosen
    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)
    return current_loss

# Define the simple gradient descent formula for training
def trainByFormula(model, inputs, outputs, learning_rate):
    current_loss = loss(model(inputs), outputs)
    Num = len(inputs)
    dw = 0
    db = 0
    # Calculating the dw and db by partial derivative
    for i in range (len(inputs)):
        dw += inputs[i]*(outputs[i]- model(inputs[i]))
        db += (outputs[i]- model(inputs[i]))
    dw = (-2/Num)*dw
    db = (-2/Num)*db
    # And here is where you update the model values based on the learning rate chosen
    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)
    return current_loss


# Define our simple linear regression model
class Model(object):
  def __init__(self):
    # Initialize the weights
    self.w = tf.Variable(INITIAL_W)
    self.b = tf.Variable(INITIAL_B)

  def __call__(self, x):
    return self.w * x + self.b


# Define our input data and learning rate
xs = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
ys = [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0]
LEARNING_RATE=0.09
# Define train action by API(1) or partial dericative formula(0)
TrainMode = 0

# Instantiate our model
model = Model()

# Collect the history of w-values and b-values to plot later
list_w, list_b = [], []
epochs = range(50)
losses = []
for epoch in epochs:
  list_w.append(model.w.numpy())
  list_b.append(model.b.numpy())
  if(TrainMode):
      current_loss = train(model, xs, ys, learning_rate=LEARNING_RATE)
  else:
      current_loss = trainByFormula(model, xs, ys, learning_rate=LEARNING_RATE)  
  losses.append(current_loss)
  print('Epoch %2d: w=%1.2f b=%1.2f, loss=%2.5f' %
        (epoch, list_w[-1], list_b[-1], current_loss))



# Plot the w-values and b-values for each training Epoch against the true values
TRUE_w = 2.0
TRUE_b = -1.0
plt.plot(epochs, list_w, 'r', epochs, list_b, 'b')
plt.plot([TRUE_w] * len(epochs), 'r--', [TRUE_b] * len(epochs), 'b--')
plt.legend(['w', 'b', 'True w', 'True b'])
plt.show()