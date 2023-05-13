from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
      # YOUR CODE HERE #
)

validation_datagen = ImageDataGenerator(
      # YOUR CODE HERE #
      rescale=1./255,
)

TRAIN_DIRECTORY_LOCATION = '2_4//train'# YOUR CODE HERE #
VAL_DIRECTORY_LOCATION = '2_4//validation'# YOUR CODE HERE #
TARGET_SIZE = (224,224) #(100,100) # YOUR CODE HERE #
CLASS_MODE = 'categorical' #'binary'# YOUR CODE HERE #

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIRECTORY_LOCATION,
    target_size = TARGET_SIZE,  
    batch_size = 128,
    class_mode = CLASS_MODE
)

validation_generator = validation_datagen.flow_from_directory(
    VAL_DIRECTORY_LOCATION,
    target_size = TARGET_SIZE,  
    batch_size = 128,
    class_mode = CLASS_MODE
)


import tensorflow as tf
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
   # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
   # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    #tf.keras.layers.Dense(1, activation='softmax')
    tf.keras.layers.Dense(3, activation='softmax')

    ## This is the first convolution
    #tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)), #(100, 100, 3)),
    #tf.keras.layers.MaxPooling2D(2, 2),
    ## The second convolution
    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    ## The third convolution
    #tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    ## The fourth convolution
    #tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    ## Flatten the results to feed into a DNN
    #tf.keras.layers.Flatten(),
    ## 512 neuron hidden layer
    #tf.keras.layers.Dense(512, activation='relu'),
    #tf.keras.layers.Dense(256, activation='relu'),
    ## Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    #tf.keras.layers.Dense(1, activation='sigmoid')
    #YOUR CODE HERE#
])

# This will print a summary of your model when you're done!
model.summary()


LOSS_FUNCTION = 'categorical_crossentropy' #'binary_crossentropy' #YOUR CODE HERE#
OPTIMIZER = 'adam' #tf.keras.optimizers.RMSprop(learning_rate=0.001) #RMSprop(lr=0.0001) #YOUR CODE HERE#

model.compile(
    loss = LOSS_FUNCTION,
    optimizer = OPTIMIZER,
    metrics = ['accuracy']
)

NUM_EPOCHS = 20 #YOUR CODE HERE#

history = model.fit(
      train_generator, 
      epochs = NUM_EPOCHS,
      verbose = 1,
      validation_data = validation_generator)

# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.xlim([0,NUM_EPOCHS])
plt.ylim([0.4,1.0])
plt.show()




