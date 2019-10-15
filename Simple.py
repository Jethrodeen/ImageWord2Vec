"""
https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8

"""
# to initialise our nn as a sequence of layers (could alternatively initialise as a graph)
from keras.models import Sequential
#for the convolution operation. Conv3D will be for videos
from keras.layers import Conv2D
# for the pooling operation
from keras.layers import MaxPooling2D
#for converting all resultant 2D arrays into single long continous linear vector
from keras.layers import Flatten
#perform the full connection of the neural network
from keras.layers import Dense


#build model
model = Sequential()

#add conv layer
"""
Four arguments:
    1. the number of filters. i.e. 32
    2. shape of each filter i.e. 3x3 
    3. input shape and type of image(RGB/Grayscale) ie. 64x64 and 3
    4. the activation function
 
"""
model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#Pooling layer.
model.add(MaxPooling2D(pool_size = (2, 2)))

#Flatten the pooled image pixels to one dimension
model.add(Flatten())

#create the fully connected (hidden) layer
"""
arguments:
    1. units = where we define the number of nodes that should be present in this hidden layer.
    2. activation function
"""
model.add(Dense(units = 128, activation= 'relu'))

#create output layer. units dependant on your class. Binary i.e. 1 in our case
model.add(Dense(units = 1, activation= 'sigmoid'))

#compile model
"""
arguments:
    1. optimizer to choose the stochastic gradient descent function
    2. Loss to choose the loss function
    3. metrics to choose performance metric
"""
model.compile(optimizer='adam', loss= 'binary_crossentropy', metrics =['accuracy'])


#checkpoint to store weights
"""
taken from 
https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0

"""
from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath="best_weights.hdf5",
                               monitor = 'val_acc',
                               verbose=1,
                               save_best_only=True)

#===============================================================================================
#image processing =https://keras.io/preprocessing/image/
#for custom dataset in folder "data"

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('data/training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
test_set = test_datagen.flow_from_directory('data/test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')


training_set.image_shape


# =============================================================================================

#train model
"""
arguments:
    1.steps per epoch: number of training images
    2.validation_steps = number of validation images
"""
history = model.fit_generator(training_set, steps_per_epoch=1589,epochs = 50, validation_data= test_set, validation_steps= 378
                    , callbacks = [checkpointer]) #this line only if we want to save best weights

#model save and accuracy plots
"""
from G Pierobon
https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0
"""
#load best weights
model.load_weights('best_weights.hdf5')

#save model
model.save('cnn.h5')

#plot
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()





#=======================================================================================================================
#Test on an image
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dog2.jfif', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = model.predict(test_image)

#check labels
training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)


#=======================================================================================================================

#check layers
from keras import models

# Extracts the outputs of all layers
layer_outputs = [layer.output for layer in model.layers]

# Creates a model that will return these outputs, given the model input
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# Returns a list of Numpy arrays: one array per layer activation
activations = activation_model.predict(test_image)

first_layer_activation = activations[2]
print(first_layer_activation.shape)