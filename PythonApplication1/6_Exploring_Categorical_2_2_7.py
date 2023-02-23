import tensorflow as tf
data = tf.keras.datasets.mnist

(training_images, training_labels), (val_images, val_labels) = data.load_data()

training_images  = training_images / 255.0
val_images = val_images / 255.0
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                    tf.keras.layers.Dense(20, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=20, validation_data=(val_images, val_labels), verbose=2)


#Examine the test data
model.evaluate(val_images, val_labels, verbose=2)

classifications = model.predict(val_images)
print(classifications[0])
print(val_labels[0])

#Modify to inspect learned values
import tensorflow as tf
data = tf.keras.datasets.mnist

(training_images, training_labels), (val_images, val_labels) = data.load_data()

training_images  = training_images / 255.0
val_images = val_images / 255.0
layer_1 = tf.keras.layers.Dense(20, activation=tf.nn.relu)
layer_2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                    layer_1,
                                    layer_2])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=20)

model.evaluate(val_images, val_labels)

classifications = model.predict(val_images)
print(classifications[0])
print(val_labels[0])

#Inspect weights
print(layer_1.get_weights()[0].size)
#The above code should print 15680. Why?
#Recall that there are 20 neurons in the first layer.
#Recall also that the images are 28x28, which is 784.
#If you multiply 784 x 20 you get 15680.
#So...this layer has 20 neurons, and each neuron learns a W parameter for each pixel. So instead of y=Mx+c, we have y=M1X1+M2X2+M3X3+....+M784X784+C in every neuron!
#Every pixel has a weight in every neuron. Those weights are multiplied by the pixel value, summed up, and given a bias.

print(layer_1.get_weights()[1].size)
#The above code will give you 20 -- the get_weights()[1] contains the biases for each of the 20 neurons in this layer.

#Inspecting layer 2
print(layer_2.get_weights()[0].size)
#This should return 200. Again, consider why?
#There are 10 neurons in this layer, but there are 20 neurons in the previous layer. So, each neuron in this layer will learn a weight for the incoming value from the previous layer. So, for example, the if the first neuron in this layer is N21, and the neurons output from the previous layers are N11-N120, then this neuron will have 20 weights (W1-W20) and it will calculate its output to be:
#W1N11+W2N12+W3N13+...+W20N120+Bias
#So each of these weights will be learned as will the bias, for every neuron.
#Note that N11 refers to Layer 1 Neuron 1.

print(layer_2.get_weights()[1].size)
#...and as expected there are 10 elements in this array, representing the 10 biases for the 10 neurons.
