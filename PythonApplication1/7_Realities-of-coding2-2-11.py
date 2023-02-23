import tensorflow as tf

data = tf.keras.datasets.mnist

(training_images, training_labels), (val_images, val_labels) = data.load_data()

training_images  = training_images / 255.0

val_images = val_images / 255.0

#Flattening the input to 784 input to the second layer(20 neurons)
#The third layer has 10 classes 
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),

                                    tf.keras.layers.Dense(20, activation=tf.nn.relu),

                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

#adam optimizer uses AdaGrad and RMSProp (Root Mean Square Propagation)
#AdaGradthat: parameters with a large gradient magnitude are assigned a smaller learning rate, 
#             while parameters with a smaller gradient magnitude are assigned a larger learning rate.
#RMSProp: adapts the learning rate of each weight parameter based on the historical gradients observed during training. 
#             However, it also introduces a moving average of the squared gradient values
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=20, 

          validation_data=(val_images, val_labels))